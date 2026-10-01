"""Grad-CAM heatmaps for Model 2: which part of the box drove the "tumour" call.

For every validation scan this saves a figure with three panels on the slice
where the heatmap peaks:
  1. CT + Grad-CAM heatmap (where the classifier looked)
  2. CT + Model 1's predicted pancreas (yellow) and tumour (red)
  3. CT + manual tumour outline (green), when one exists (MSD scans)

It also scores the heatmaps against the manual tumour outlines: a "hit" means
the heatmap's peak lies within 10 mm of the real tumour. That checks whether
the classifier actually looks at the tumour or at something else.

Grad-CAM is a coarse attention map (the last conv layer is 1/16 resolution),
NOT a tumour outline. Model 1's red tumour mask is the actual outline.

Outputs: outputs/classification/<tag>/gradcam/<case>.png, gradcam_hits.json
Usage: python src/gradcam.py [--tag pancreas]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from cls_train import INPUT, Net, prepare
from common import HU_MAX, HU_MIN, OUTPUTS, device, load_cases
from crops import CROP, crop_case
from split import get_split

OFF = [(c - i) // 2 for c, i in zip(CROP, INPUT)]  # eval input = centre of the crop
CENTRE = tuple(slice(o, o + i) for o, i in zip(OFF, INPUT))


def gradcam(model, x):
    """x: 1 x C x INPUT (already prepared). Returns (prob, cam in [0,1] at INPUT resolution)."""
    acts = {}
    last_relu = [m for m in model.features if isinstance(m, torch.nn.ReLU)][-1]
    h = last_relu.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    model.eval()
    x = x.clone().requires_grad_(False)
    logit = model(x)
    h.remove()
    a = acts["a"]
    g, = torch.autograd.grad(logit.sum(), a)
    cam = F.relu((g.mean((2, 3, 4), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=INPUT, mode="trilinear", align_corners=False)[0, 0]
    cam = cam / cam.max().clamp(min=1e-8)
    return torch.sigmoid(logit).item(), cam.detach().cpu().numpy()


def load_classifier(tag, dev):
    state = torch.load(OUTPUTS / "classification" / tag / "model.pt", map_location=dev)
    model = Net(state["features.0.weight"].shape[1]).to(dev)
    model.load_state_dict(state)
    return model.eval()


def figure(cid, label, prob, ct, cam, organ, tumor, gt, path):
    z = int(np.unravel_index(cam.argmax(), cam.shape)[0])  # slice where the heatmap peaks
    win = np.clip(ct[z], HU_MIN, HU_MAX)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(win, cmap="gray"); ax[0].imshow(cam[z], cmap="jet", alpha=0.4, vmin=0, vmax=1)
    ax[0].set_title(f"{cid} ({'tumour' if label else 'healthy'}): P(tumour) = {prob:.2f}\nGrad-CAM: where Model 2 looked", fontsize=9)
    ax[1].imshow(win, cmap="gray")
    if organ[z].any():
        ax[1].contour(organ[z], [0.5], colors="yellow", linewidths=1)
    if tumor[z].any():
        ax[1].imshow(np.ma.masked_where(~tumor[z], tumor[z]), cmap="autumn", alpha=0.5)
        ax[1].contour(tumor[z], [0.5], colors="red", linewidths=1.2)
    ax[1].set_title(f"Model 1: pancreas (yellow), tumour (red)\npredicted tumour {tumor.sum() * 1.5 ** 3 / 1000:.1f} ml", fontsize=9)
    ax[2].imshow(win, cmap="gray")
    if gt is not None and gt[z].any():
        ax[2].contour(gt[z], [0.5], colors="lime", linewidths=1.2)
    ax[2].set_title("manual tumour outline (green)" if gt is not None and gt.any() else "no manual tumour outline", fontsize=9)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=85); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pancreas")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    dev = device()
    model = load_classifier(args.tag, dev)
    n_ch = model.features[0].in_channels
    out = OUTPUTS / "classification" / args.tag / "gradcam"
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases().set_index("case_id")

    hits = []
    for cid in get_split()[args.split]:
        chans = ("ct", "predpancreas")[:n_ch]
        raw = crop_case(cid, "pancreas", chans + ("predpancreas", "predtumor", "tumor"))
        x = torch.from_numpy(raw[:n_ch]).float()[None].to(dev)
        if n_ch == 1:
            x = torch.cat([x, torch.zeros_like(x)], 1)
        x = prepare(x, False, 1.0)[:, :n_ch]
        prob, cam = gradcam(model, x)
        ct, organ, tumor, gt = (raw[i][CENTRE].astype(np.float32) for i in (0, -3, -2, -1))
        organ, tumor, gt = organ > 0.5, tumor > 0.5, gt > 0.5
        label = int(cases.label[cid])
        figure(cid, label, prob, ct, cam, organ, tumor, gt if gt.any() else None, out / f"{cid}.png")
        if gt.any():
            peak = np.unravel_index(cam.argmax(), cam.shape)
            dist = ndimage.distance_transform_edt(~gt, sampling=1.5)[peak]
            hits.append(dict(case_id=cid, prob=prob, peak_mm_from_tumour=float(dist), hit=bool(dist <= 10)))
    summary = dict(n_scans_with_tumour_outline=len(hits),
                   hit_rate_within_10mm=float(np.mean([h["hit"] for h in hits])) if hits else None,
                   median_peak_mm_from_tumour=float(np.median([h["peak_mm_from_tumour"] for h in hits])) if hits else None,
                   cases=hits)
    (OUTPUTS / "classification" / args.tag / "gradcam_hits.json").write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "cases"})


if __name__ == "__main__":
    main()
