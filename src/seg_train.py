"""Model 1 of 2: 3D U-Net that outlines the pancreas AND any tumour.

Classes: 0 background, 1 pancreas, 2 tumour.
Trained on every scan that has manual outlines:
  * Pancreas-CT healthy scans  (pancreas outlined, no tumour)
  * MSD Task07 tumour scans    (pancreas + tumour outlined)
(The Pancreatic-CT-CBCT-SEG cancer scans have no outlines, so they're only used
by the classifier.) Uses the shared 85/15 split from src/split.py.

Differences from the legacy 2D segmentationCode.py:
  * 3D context instead of independent 2D slices
  * Dice + cross-entropy loss (plain BCE on a ~0.5%-of-voxels organ collapsed to
    "predict background everywhere", which is why the old Dice was ~0.02)
  * patches sampled with extra weight on pancreas and tumour (1:1:2)
  * images & masks loaded as matched file pairs (the old CSV loader sorted image
    paths and mask paths independently, which can pair the wrong mask)
  * HU windowing + normalisation, no left/right flips (anatomy is asymmetric)
  * train/val loss and Dice (pancreas + tumour) tracked; best-val checkpoint kept

Outputs (outputs/segmentation/): model.pt, history.json, learning_curves.png
Usage: python src/seg_train.py [--epochs 120 --iters 50]
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
from monai.data import Dataset, ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai import transforms as T

from common import HU_MAX, HU_MIN, OUTPUTS, PROC, device, load_cases, seed_all
from split import get_split

PATCH = (96, 96, 96)
OUT = OUTPUTS / "segmentation"
CLASSES = {1: "pancreas", 2: "tumor"}


def build_model():
    return UNet(spatial_dims=3, in_channels=1, out_channels=3, channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2), num_res_units=2, norm="instance", dropout=0.1)


def labelled_cases():
    """Cases with manual outlines; writes a combined 0/1/2 label file for each (once)."""
    ids = []
    for c in load_cases().itertuples():
        if not (PROC / f"{c.case_id}_pancreas.nii.gz").exists():
            continue
        out = PROC / f"{c.case_id}_seglabel.nii.gz"
        if not out.exists():
            pan = sitk.ReadImage(str(PROC / f"{c.case_id}_pancreas.nii.gz"))
            arr = (sitk.GetArrayFromImage(pan) > 0).astype(np.uint8)
            tum = PROC / f"{c.case_id}_tumor.nii.gz"
            if tum.exists():
                arr[sitk.GetArrayFromImage(sitk.ReadImage(str(tum))) > 0] = 2
            img = sitk.GetImageFromArray(arr)
            img.CopyInformation(pan)
            sitk.WriteImage(img, str(out), useCompression=True)
        ids.append(c.case_id)
    return ids


def class_dice(logits, y):
    """Per-class Dice of the argmax prediction over the whole batch; None if the class is absent in both."""
    p, t = logits.argmax(1), y[:, 0].long()
    out = {}
    for k, name in CLASSES.items():
        pk, tk = p == k, t == k
        denom = (pk.sum() + tk.sum()).item()
        out[name] = None if denom == 0 else 2 * (pk & tk).sum().item() / denom
    return out


def plot_history(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, key, title in [(ax[0], "loss", "Loss"), (ax[1], "dice_pancreas", "Pancreas Dice"), (ax[2], "dice_tumor", "Tumour Dice")]:
        a.plot(ep, [h[f"train_{key}"] for h in hist], color="#1f77b4", label="train")
        v = [(h["epoch"], h[f"val_{key}"]) for h in hist if h.get(f"val_{key}") is not None]
        if v:
            a.plot(*zip(*v), color="#d62728", marker="o", ms=3, label="validation")
        a.set_title(f"Segmentation {title.lower()}"); a.set_xlabel("epoch"); a.set_ylabel(title); a.grid(alpha=.3); a.legend()
        if "Dice" in title:
            a.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--iters", type=int, default=50, help="training iterations per epoch")
    ap.add_argument("--val_every", type=int, default=10)
    args = ap.parse_args()
    seed_all(0)
    dev = device()
    OUT.mkdir(parents=True, exist_ok=True)

    split, labelled = get_split(), set(labelled_cases())
    item = lambda c: {"image": str(PROC / f"{c}_ct.nii.gz"), "label": str(PROC / f"{c}_seglabel.nii.gz")}
    train = [item(c) for c in split["train"] if c in labelled]
    val = [item(c) for c in split["val"] if c in labelled]
    print(f"segmentation: {len(train)} train / {len(val)} val labelled scans")

    load = [T.LoadImaged(["image", "label"], image_only=True), T.EnsureChannelFirstd(["image", "label"]),
            T.ScaleIntensityRanged("image", HU_MIN, HU_MAX, 0.0, 1.0, clip=True)]
    train_tf = T.Compose(load + [
        T.SpatialPadd(["image", "label"], PATCH),
        T.RandCropByLabelClassesd(["image", "label"], "label", PATCH, ratios=[1, 1, 2], num_classes=3,
                                  num_samples=2, warn=False),
        T.RandAffined(["image", "label"], prob=0.3, rotate_range=(0.26,) * 3, scale_range=(0.15,) * 3,
                      mode=("bilinear", "nearest"), padding_mode="border"),
        T.RandGaussianNoised("image", prob=0.15, std=0.02),
        T.RandGaussianSmoothd("image", prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
        T.RandScaleIntensityd("image", 0.1, prob=0.3), T.RandShiftIntensityd("image", 0.1, prob=0.3),
        T.RandAdjustContrastd("image", prob=0.15, gamma=(0.7, 1.5)),
    ])
    # ~300 volumes don't fit in 8 GB RAM, so they're read from disk in a background thread
    tl = ThreadDataLoader(Dataset(train, train_tf), batch_size=2, shuffle=True, num_workers=0, buffer_size=4)
    vds = Dataset(val, T.Compose(load))

    model = build_model().to(dev)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False, batch=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.epochs * args.iters, pct_start=0.05)

    def batches():
        while True:
            yield from tl
    stream = batches()
    hist, best = [], -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, losses, dices = time.time(), [], []
        for _ in range(args.iters):
            b = next(stream)
            x, y = b["image"].to(dev), b["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
            dices.append(class_dice(out.detach(), y))
        h = dict(epoch=ep, train_loss=float(np.mean(losses)))
        for name in CLASSES.values():
            h[f"train_dice_{name}"] = mean_or_none([d[name] for d in dices])

        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            vl, vd = [], []
            with torch.no_grad():
                for v in vds:
                    x, y = v["image"][None].to(dev), v["label"][None].to(dev)
                    out = sliding_window_inference(x, PATCH, 4, model, overlap=0.25)
                    vl.append(loss_fn(out, y).item())
                    vd.append(class_dice(out, y))
            h["val_loss"] = float(np.mean(vl))
            for name in CLASSES.values():
                h[f"val_dice_{name}"] = mean_or_none([d[name] for d in vd])
            score = np.mean([h["val_dice_pancreas"] or 0, h["val_dice_tumor"] or 0])
            if score > best:
                best = score
                torch.save(model.state_dict(), OUT / "model.pt")
        hist.append(h)
        print(" ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in h.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        (OUT / "history.json").write_text(json.dumps(hist, indent=1))
        plot_history(hist, OUT / "learning_curves.png")
    print(f"best val mean(pancreas, tumour) Dice {best:.4f}")


if __name__ == "__main__":
    main()
