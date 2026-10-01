"""Model 1 of 2: 3D U-Net that segments the pancreas (Pancreas-CT manual labels).

Uses the shared 85/15 split (src/split.py): trained on the healthy scans in the
train split, validated on the healthy scans in the val split. (Cancer scans have
no pancreas labels, so they can't be used to train or score segmentation.)

Differences from the legacy 2D segmentationCode.py:
  * 3D context instead of independent 2D slices
  * Dice + cross-entropy loss (plain BCE on a ~0.5%-of-voxels organ collapsed to
    "predict background everywhere", which is why the old Dice was ~0.02)
  * patches sampled around the pancreas (pos:neg = 2:1)
  * images & masks loaded as matched file pairs (the old CSV loader sorted image
    paths and mask paths independently, which can pair the wrong mask)
  * HU windowing + normalisation, no left/right flips (anatomy is asymmetric)
  * Dice tracked on train and val every epoch; best-val-Dice checkpoint kept

Outputs (outputs/segmentation/): model.pt, history.json, learning_curves.png
Usage: python src/seg_train.py [--epochs 100]
"""
import argparse
import json
import time

import torch
from monai.data import CacheDataset, ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai import transforms as T

from common import HU_MAX, HU_MIN, OUTPUTS, PROC, device, load_cases, seed_all
from plots import learning_curves
from split import get_split

PATCH = (96, 96, 96)
OUT = OUTPUTS / "segmentation"


def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.1)


def base_transforms():
    keys = ["image", "label"]
    return [
        T.LoadImaged(keys, image_only=True), T.EnsureChannelFirstd(keys),
        T.ScaleIntensityRanged("image", HU_MIN, HU_MAX, 0.0, 1.0, clip=True),
        # float16 cache keeps ~70 volumes inside an 8 GB laptop's RAM
        T.EnsureTyped(keys, dtype=[torch.float16, torch.uint8], track_meta=False),
    ]


def hard_dice(logits, y):
    """Dice of argmax prediction vs label, summed over the batch (one number per batch)."""
    p = logits.argmax(1) == 1
    t = y[:, 0] > 0
    return (2 * (p & t).sum() / ((p.sum() + t.sum()).clamp(min=1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val_every", type=int, default=5)
    args = ap.parse_args()
    seed_all(0)
    dev = device()
    OUT.mkdir(parents=True, exist_ok=True)

    split, healthy = get_split(), set(load_cases().query("label == 0").case_id)
    item = lambda c: {"image": str(PROC / f"{c}_ct.nii.gz"), "label": str(PROC / f"{c}_pancreas.nii.gz")}
    train = [item(c) for c in split["train"] if c in healthy]
    val = [item(c) for c in split["val"] if c in healthy]
    print(f"segmentation: {len(train)} train / {len(val)} val healthy scans")

    train_tf = T.Compose(base_transforms() + [
        T.EnsureTyped(["image", "label"], dtype=torch.float32, track_meta=False),
        T.SpatialPadd(["image", "label"], PATCH),
        T.RandCropByPosNegLabeld(["image", "label"], "label", PATCH, pos=2, neg=1, num_samples=2),
        T.RandAffined(["image", "label"], prob=0.3, rotate_range=(0.26,) * 3, scale_range=(0.15,) * 3,
                      mode=("bilinear", "nearest"), padding_mode="border"),
        T.RandGaussianNoised("image", prob=0.15, std=0.02),
        T.RandGaussianSmoothd("image", prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        T.RandScaleIntensityd("image", 0.1, prob=0.3), T.RandShiftIntensityd("image", 0.1, prob=0.3),
        T.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
    ])
    # threads, not processes: macOS worker processes would each copy the whole cache
    tl = ThreadDataLoader(CacheDataset(train, train_tf, num_workers=4), batch_size=2, shuffle=True, num_workers=0)
    vds = CacheDataset(val, T.Compose(base_transforms()), num_workers=4)

    model = build_model().to(dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * len(tl), pct_start=0.05)

    hist, best = [], -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot_loss, tot_dice = time.time(), 0.0, 0.0
        for b in tl:
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item()
            tot_dice += hard_dice(out.detach(), y)
        h = dict(epoch=ep, train_loss=tot_loss / len(tl), train_metric=tot_dice / len(tl), val_loss=None, val_metric=None)

        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            vl, vd = 0.0, 0.0
            with torch.no_grad():
                for v in vds:
                    x, y = v["image"][None].float().to(dev), v["label"][None].to(dev)
                    out = sliding_window_inference(x, PATCH, 4, model, overlap=0.25)
                    vl += loss_fn(out, y).item()
                    vd += hard_dice(out, y)
            h["val_loss"], h["val_metric"] = vl / len(vds), vd / len(vds)
            if h["val_metric"] > best:
                best = h["val_metric"]
                torch.save(model.state_dict(), OUT / "model.pt")
        hist.append(h)
        msg = f"ep {ep} train loss {h['train_loss']:.4f} dice {h['train_metric']:.3f}"
        if h["val_loss"] is not None:
            msg += f" | val loss {h['val_loss']:.4f} dice {h['val_metric']:.3f}"
        print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)
        (OUT / "history.json").write_text(json.dumps(hist, indent=1))
        learning_curves(hist, OUT / "learning_curves.png", "Dice", "Segmentation")
    print(f"best val Dice {best:.4f}")


if __name__ == "__main__":
    main()
