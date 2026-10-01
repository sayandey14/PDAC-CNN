"""Train a 3D U-Net to segment the pancreas (Pancreas-CT manual labels).

Trained with 5-fold cross-validation so every healthy scan gets an
*out-of-fold* prediction later. If healthy scans were localised by a model that
had seen their labels, while cancer scans were not, the crops would differ in
quality between the two classes, which is a bias the classifier could exploit.

Differences from the legacy 2D segmentationCode.py:
  * 3D context instead of independent 2D slices
  * Dice + cross-entropy loss (plain BCE on a ~0.5%-of-voxels organ collapsed to
    "predict background everywhere", which is why the old Dice was ~0.02)
  * patches sampled around the pancreas (pos:neg = 2:1)
  * images & masks loaded together from one file pair (the old CSV loader sorted
    image paths and mask paths independently, which can pair the wrong mask)
  * proper HU windowing + normalisation, no left/right flips (anatomy is asymmetric)

Usage: python src/seg_train.py --fold 0 [--epochs 300]
"""
import argparse
import json
import time

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai import transforms as T
from sklearn.model_selection import KFold

from common import HU_MAX, HU_MIN, OUTPUTS, PROC, device, load_cases, seed_all

PATCH = (96, 96, 96)
N_FOLDS = 5


def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.1)


def base_transforms(with_label=True):
    keys = ["image", "label"] if with_label else ["image"]
    return [
        T.LoadImaged(keys, image_only=True), T.EnsureChannelFirstd(keys),
        T.ScaleIntensityRanged("image", HU_MIN, HU_MAX, 0.0, 1.0, clip=True),
    ]


def folds():
    path = OUTPUTS / "seg_folds.json"
    if path.exists():
        return json.loads(path.read_text())
    ids = sorted(load_cases().query("label == 0").case_id)
    f = {cid: int(k) for k, (_, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=0).split(ids)) for cid in np.array(ids)[te]}
    OUTPUTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(f, indent=1))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--val_every", type=int, default=20)
    args = ap.parse_args()
    seed_all(args.fold)
    dev = device()

    fmap = folds()
    item = lambda c: {"image": str(PROC / f"{c}_ct.nii.gz"), "label": str(PROC / f"{c}_pancreas.nii.gz"), "id": c}
    train = [item(c) for c, k in fmap.items() if k != args.fold]
    val = [item(c) for c, k in fmap.items() if k == args.fold]

    train_tf = T.Compose(base_transforms() + [
        T.SpatialPadd(["image", "label"], PATCH),
        T.RandCropByPosNegLabeld(["image", "label"], "label", PATCH, pos=2, neg=1, num_samples=2),
        T.RandAffined(["image", "label"], prob=0.3, rotate_range=(0.26,) * 3, scale_range=(0.15,) * 3,
                      mode=("bilinear", "nearest"), padding_mode="border"),
        T.RandGaussianNoised("image", prob=0.15, std=0.02),
        T.RandGaussianSmoothd("image", prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        T.RandScaleIntensityd("image", 0.1, prob=0.3), T.RandShiftIntensityd("image", 0.1, prob=0.3),
        T.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
    ])
    val_tf = T.Compose(base_transforms())

    tl = DataLoader(CacheDataset(train, train_tf, num_workers=4), batch_size=2, shuffle=True, num_workers=2,
                    persistent_workers=True)
    vds = CacheDataset(val, val_tf, num_workers=4)

    model = build_model().to(dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * len(tl), pct_start=0.05)
    dice = DiceMetric(include_background=False)

    log = OUTPUTS / f"seg_fold{args.fold}.log"
    best = -1
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for b in tl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"ep {ep} loss {tot / len(tl):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % args.val_every == 0 or ep == args.epochs - 1:
            model.eval()
            dice.reset()
            with torch.no_grad():
                for v in vds:
                    x = v["image"][None].to(dev)
                    p = sliding_window_inference(x, PATCH, 2, model, overlap=0.25).argmax(1, keepdim=True)
                    dice(torch.nn.functional.one_hot(p[:, 0].long(), 2).permute(0, 4, 1, 2, 3).cpu(),
                         torch.nn.functional.one_hot(v["label"][None, 0].long(), 2).permute(0, 4, 1, 2, 3))
            d = dice.aggregate().item()
            msg += f" | val dice {d:.4f}"
            if d > best:
                best = d
                torch.save(model.state_dict(), OUTPUTS / f"seg_fold{args.fold}.pt")
                msg += " *"
        print(msg, flush=True)
        with open(log, "a") as f:
            f.write(msg + "\n")
    print(f"fold {args.fold} best val dice {best:.4f}")


if __name__ == "__main__":
    main()
