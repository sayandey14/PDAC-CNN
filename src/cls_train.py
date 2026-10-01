"""Model 2 of 2: cancer vs. healthy classifier, a 3D CNN on pancreas-centred crops.

Input: a fixed 1.5 mm box (96 x 144 x 192 mm) centred on the pancreas that
Model 1 predicted, 2 channels: CT + predicted pancreas mask.

Uses the shared 85/15 patient split (src/split.py). Every epoch it logs loss and
accuracy on the training set (no augmentation) and on the validation set, then
writes learning curves, confusion matrices (TP/FP/FN/TN), ROC and PR curves and
a probability histogram. The reported model is the FINAL epoch. The validation
set is never used to choose a checkpoint, so its numbers stay honest. (The
legacy code saved whichever epoch scored best on the 32 validation scans and
then reported that same score, which inflates it.)

Other changes from legacy myclassifier.py:
  * real 3D network (the old GoogLeNet was 2D with the slices fed in as channels)
  * pancreas-centred box + mask channel instead of the whole resized scan
  * GPU augmentation: rotation, scaling, shifts, intensity, noise
  * class-weighted loss (healthy:cancer = 2:1), AdamW + one-cycle LR, label smoothing
  * label convention: 1 = cancer (the old code used 0 = cancer)

Note: healthy *training* scans are localised by a segmentation model that saw
their labels, so their boxes may be centred slightly better than cancer boxes.
Random shifts of up to ±12 mm during training wash this out, and validation
scans of both classes are localised identically (unseen by Model 1).

Outputs: outputs/classification/<tag>/
Usage: python src/cls_train.py [--region pancreas|control] [--no_mask] [--epochs 150]
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HU_MAX, HU_MIN, OUTPUTS, ROOT, device, load_cases, seed_all
from make_crops import CROP
from plots import confusion_matrix_plot, learning_curves, prob_histogram, roc_pr_plot, summary_metrics
from split import get_split

INPUT = (64, 96, 128)
THRESHOLD = 0.5


class Net(nn.Module):
    """Small 3D CNN: stride-2 stem, 4 stages of (conv-BN-ReLU) x2 + downsample, global pooling, dropout, linear."""

    def __init__(self, in_ch, width=24, drop=0.4):
        super().__init__()
        layers = [nn.Conv3d(in_ch, width, 3, stride=2, padding=1, bias=False), nn.BatchNorm3d(width), nn.ReLU(inplace=True)]
        c = width
        for i, w in enumerate([width, width * 2, width * 4, width * 8]):
            layers += [nn.Conv3d(c, w, 3, padding=1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True),
                       nn.Conv3d(w, w, 3, padding=1, bias=False), nn.BatchNorm3d(w), nn.ReLU(inplace=True)]
            if i < 3:
                layers.append(nn.MaxPool3d(2))
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(c * 2, 1))

    def forward(self, x):
        f = self.features(x)
        return self.head(torch.cat([f.mean((2, 3, 4)), f.amax((2, 3, 4))], 1))[:, 0]


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    r = int(math.ceil(2 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    c = x.shape[1]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = -1
        pad = [0] * 6
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape).expand(c, 1, *k.view(shape).shape[2:]), groups=c)
    return x


def prepare(batch, train, blur):
    """batch: N x 2 x CROP (raw HU, mask) -> N x C x INPUT, normalised, optionally augmented."""
    n = batch.shape[0]
    ct = (batch[:, :1].clamp(HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
    x = torch.cat([ct, batch[:, 1:]], 1)
    frac = torch.tensor([INPUT[2] / CROP[2], INPUT[1] / CROP[1], INPUT[0] / CROP[0]], device=x.device)  # x,y,z
    theta = torch.zeros(n, 3, 4, device=x.device)
    theta[:, :3, :3] = torch.diag(frac)
    if train:
        ang = torch.stack([torch.empty(n).uniform_(-a, a) for a in (0.26, 0.09, 0.09)], 1).to(x.device)  # axial ±15°, others ±5°
        R = []
        for az, ay, ax in ang.tolist():
            cz, sz, cy, sy, cx, sx = math.cos(az), math.sin(az), math.cos(ay), math.sin(ay), math.cos(ax), math.sin(ax)
            Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])  # rotation in the axial (x-y) plane
            Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            R.append(Rz @ Ry @ Rx)
        R = torch.stack(R).to(x.device)
        scale = torch.empty(n, 1, 1, device=x.device).uniform_(0.9, 1.1)
        theta[:, :3, :3] = R @ torch.diag(frac) * scale
        max_shift = 1 - frac  # stay within the larger crop
        theta[:, :3, 3] = (torch.rand(n, 3, device=x.device) * 2 - 1) * max_shift
    grid = F.affine_grid(theta, (n, x.shape[1], *INPUT), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    ct = gaussian_blur(x[:, :1], blur)  # always on: suppresses scanner-specific noise texture
    if train:
        ct = ct * torch.empty(n, 1, 1, 1, 1, device=x.device).uniform_(0.9, 1.1) \
             + torch.empty(n, 1, 1, 1, 1, device=x.device).uniform_(-0.1, 0.1)
        ct = ct + torch.randn_like(ct) * torch.empty(n, 1, 1, 1, 1, device=x.device).uniform_(0, 0.03)
    return torch.cat([ct, x[:, 1:]], 1)




def load_crops(region, ids, mask=True):
    d = ROOT / "data" / "crops" / region
    X = torch.from_numpy(np.stack([np.load(d / f"{c}.npy") for c in ids]))
    return X if mask else X[:, :1]


def to_input(xb, n_ch, train, blur, dev):
    xb = xb.to(dev).float()
    if xb.shape[1] == 1:
        xb = torch.cat([xb, torch.zeros_like(xb)], 1)
    return prepare(xb, train, blur)[:, :n_ch]


@torch.no_grad()
def predict(model, X, n_ch, blur, dev, bs=16):
    model.eval()
    return torch.cat([torch.sigmoid(model(to_input(X[i:i + bs], n_ch, False, blur, dev))) for i in range(0, len(X), bs)]).cpu().numpy()


def bce(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="pancreas")
    ap.add_argument("--no_mask", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.region}{'_nomask' if args.no_mask else ''}"
    out = OUTPUTS / "classification" / tag
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    dev = device()

    split, labels = get_split(), load_cases().set_index("case_id").label
    ids = {s: split[s] for s in ("train", "val")}
    X = {s: load_crops(args.region, ids[s], not args.no_mask) for s in ids}
    y = {s: labels[ids[s]].values.astype(np.float32) for s in ids}
    n_ch = X["train"].shape[1]
    print(f"classifier [{tag}]: train {len(y['train'])} ({int(y['train'].sum())} cancer), "
          f"val {len(y['val'])} ({int(y['val'].sum())} cancer)")

    model = Net(n_ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = len(y["train"]) // args.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    pos_weight = torch.tensor((y["train"] == 0).sum() / (y["train"] == 1).sum(), device=dev)

    hist = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(y["train"]))
        for i in range(steps_per_epoch):  # drop the last partial batch (BatchNorm dislikes tiny batches)
            idx = perm[i * args.bs:(i + 1) * args.bs]
            xb = to_input(X["train"][idx], n_ch, True, args.blur, dev)
            yb = torch.from_numpy(y["train"][idx]).to(dev)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb * 0.9 + 0.05, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        # both sets scored the same way: eval mode, no augmentation, plain (unweighted) BCE
        h = dict(epoch=ep)
        for s in ("train", "val"):
            p = predict(model, X[s], n_ch, args.blur, dev)
            h[f"{s}_loss"], h[f"{s}_metric"] = bce(p, y[s]), float(((p >= THRESHOLD) == y[s]).mean())
        hist.append(h)
        print(f"ep {ep} train loss {h['train_loss']:.4f} acc {h['train_metric']:.3f} | "
              f"val loss {h['val_loss']:.4f} acc {h['val_metric']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if ep % 5 == 0 or ep == args.epochs:
            learning_curves(hist, out / "learning_curves.png", "Accuracy", "Classifier")
            (out / "history.json").write_text(json.dumps(hist, indent=1))

    torch.save(model.state_dict(), out / "model.pt")
    probs = {s: predict(model, X[s], n_ch, args.blur, dev) for s in ("train", "val")}
    results = {}
    for s in ("train", "val"):
        results[s] = summary_metrics(y[s], probs[s], THRESHOLD)
        confusion_matrix_plot(results[s], out / f"confusion_{s}.png",
                              f"{s.capitalize()} set (n={len(y[s])}), acc {results[s]['accuracy']:.1%}")
    roc_pr_plot({s: (y[s], probs[s]) for s in ("train", "val")}, out / "roc_pr.png")
    prob_histogram(y["val"], probs["val"], out / "val_probabilities.png", "Validation set predictions", THRESHOLD)
    pd.concat([pd.DataFrame(dict(case_id=ids[s], split=s, label=y[s].astype(int), prob_cancer=probs[s],
                                 predicted=(probs[s] >= THRESHOLD).astype(int))) for s in ("train", "val")]) \
        .to_csv(out / "predictions.csv", index=False)
    results["config"] = vars(args)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=float))

    print("\n            " + "  ".join(f"{k:>9}" for k in ["TP", "FP", "FN", "TN", "accuracy", "sens", "spec", "AUC"]))
    for s in ("train", "val"):
        r = results[s]
        print(f"{s:>10}  " + "  ".join(f"{v:>9}" for v in [r["TP"], r["FP"], r["FN"], r["TN"]])
              + "  " + "  ".join(f"{v:>9.3f}" for v in [r["accuracy"], r["sensitivity_recall"], r["specificity"], r["auc"]]))


if __name__ == "__main__":
    main()
