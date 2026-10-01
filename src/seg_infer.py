"""Run Model 1 (pancreas + tumour segmentation) on every scan, evaluate it, save QC.

Predicted pancreas masks feed Model 2 (the classifier's box is centred on
them). Predicted tumour masks are the "where is the tumour" output.

Evaluation (outputs/segmentation/), all reported separately for train and val:
  metrics.json             pancreas Dice, tumour Dice, voxel TP/FP/FN/TN, and
                           scan-level tumour detection (TP/FP/FN/TN, sens/spec)
                           where "tumour found" = predicted tumour >= 0.5 ml
  dice_per_case.png        validation pancreas & tumour Dice per scan
  confusion_detection_val.png   scan-level tumour detection on validation
  voxel_confusion_val.png       voxel-level tumour confusion on validation
  qc/<case>.png            overlays: yellow = predicted pancreas, red = predicted
                           tumour, green = manual outline (tumour if present,
                           else pancreas; stomach/duodenum for CBCT-SEG cases)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

from common import HU_MAX, HU_MIN, PROC, device, load_cases
from plots import confusion_counts, confusion_matrix_plot
from seg_train import OUT, PATCH, build_model
from split import get_split

MIN_TUMOR_ML = 0.5  # smaller predicted "tumours" are treated as noise
VOX_ML = 1.5 ** 3 / 1000


def load_model(dev):
    m = build_model().to(dev).eval()
    m.load_state_dict(torch.load(OUT / "model.pt", map_location=dev))
    return m


def keep_main_components(m, frac=0.2):
    """Drop specks: keep connected components >= frac of the largest one."""
    lab, n = ndimage.label(m)
    if n == 0:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.where(sizes >= frac * sizes.max())[0])


def segment(model, ct_hu: np.ndarray, dev):
    """Returns (pancreas_incl_tumour, tumour) boolean masks."""
    x = torch.from_numpy((np.clip(ct_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).float()[None, None].to(dev)
    with torch.no_grad():
        cls = sliding_window_inference(x, PATCH, 4, model, overlap=0.5, mode="gaussian").argmax(1)[0].cpu().numpy()
    organ = keep_main_components(cls >= 1)
    tumor = (cls == 2) & organ
    lab, n = ndimage.label(tumor)
    if n:
        sizes = ndimage.sum(tumor, lab, range(1, n + 1)) * VOX_ML
        tumor = np.isin(lab, 1 + np.where(sizes >= MIN_TUMOR_ML)[0])
    return organ, tumor


def overlay(ax, ct2d, organ, tumor, ref=None):
    ax.imshow(np.clip(ct2d, HU_MIN, HU_MAX), cmap="gray")
    if organ.any():
        ax.contour(organ, [0.5], colors="yellow", linewidths=1)
    if tumor.any():
        ax.imshow(np.ma.masked_where(~tumor, tumor), cmap="autumn", alpha=0.5)
        ax.contour(tumor, [0.5], colors="red", linewidths=1.2)
    if ref is not None and ref.any():
        ax.contour(ref, [0.5], colors="lime", linewidths=1, linestyles="--")
    ax.axis("off")


def qc_png(cid, ct, organ, tumor, ref, path, title):
    focus = tumor if tumor.any() else organ if organ.any() else np.ones_like(organ)
    zz, yy, _ = np.where(focus)
    z, y = int(np.median(zz)), int(np.median(yy))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    overlay(ax[0], ct[z], organ[z], tumor[z], None if ref is None else ref[z])
    f = lambda a: a[:, y][::-1]
    overlay(ax[1], f(ct), f(organ), f(tumor), None if ref is None else f(ref))
    ax[0].set_title(f"{cid} axial — {title}", fontsize=9)
    ax[1].set_title("coronal  (yellow pancreas, red predicted tumour, green dashed manual)", fontsize=8)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def dice(a, b):
    s = a.sum() + b.sum()
    return None if s == 0 else 2 * (a & b).sum() / s


def read_mask(cid, name):
    p = PROC / f"{cid}_{name}.nii.gz"
    return sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0 if p.exists() else None


def main():
    dev = device()
    model = load_model(dev)
    split = get_split()
    part = {c: "train" for c in split["train"]} | {c: "val" for c in split["val"]}
    (OUT / "qc").mkdir(parents=True, exist_ok=True)

    rows = []
    for c in load_cases().itertuples():
        img = sitk.ReadImage(str(PROC / f"{c.case_id}_ct.nii.gz"))
        ct = sitk.GetArrayFromImage(img).astype(np.float32)
        organ, tumor = segment(model, ct, dev)
        for name, m in [("predpancreas", organ), ("predtumor", tumor)]:
            out = sitk.GetImageFromArray(m.astype(np.uint8))
            out.CopyInformation(img)
            sitk.WriteImage(out, str(PROC / f"{c.case_id}_{name}.nii.gz"), useCompression=True)

        row = dict(case_id=c.case_id, label=c.label, source=c.source, split=part[c.case_id],
                   pred_pancreas_ml=organ.sum() * VOX_ML, pred_tumor_ml=tumor.sum() * VOX_ML)
        row["tumor_found"] = int(row["pred_tumor_ml"] >= MIN_TUMOR_ML)
        gt_organ, gt_tumor = read_mask(c.case_id, "pancreas"), read_mask(c.case_id, "tumor")
        if c.label == 0:
            gt_tumor = np.zeros_like(tumor)  # healthy: no tumour by definition
        ref = None
        if gt_organ is not None:
            row["dice_pancreas"] = dice(organ, gt_organ)
            ref = gt_organ
        if gt_tumor is not None:
            row["dice_tumor"] = dice(tumor, gt_tumor)
            row.update(vTP=int((tumor & gt_tumor).sum()), vFP=int((tumor & ~gt_tumor).sum()),
                       vFN=int((~tumor & gt_tumor).sum()), vTN=int((~tumor & ~gt_tumor).sum()))
            if gt_tumor.any():
                ref = gt_tumor
        if ref is None:
            ref = read_mask(c.case_id, "stomduo")
        title = f"{row['split']}, label {'tumour' if c.label else 'healthy'}, pred tumour {row['pred_tumor_ml']:.1f} ml"
        if row.get("dice_tumor") is not None:
            title += f", tumour Dice {row['dice_tumor']:.2f}"
        qc_png(c.case_id, ct, organ, tumor, ref, OUT / "qc" / f"{c.case_id}.png", title)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_case.csv", index=False)

    summary = {}
    for s in ["train", "val"]:
        d = df[df.split == s]
        det = confusion_counts(d.label, d.tumor_found)
        vox = {k: int(d[f"v{k}"].sum()) for k in ["TP", "FP", "FN", "TN"]}
        summary[s] = dict(
            pancreas_dice_mean=d.dice_pancreas.mean(), pancreas_dice_median=d.dice_pancreas.median(),
            tumor_dice_mean=d[d.source == "MSD-Task07"].dice_tumor.mean(),
            tumor_dice_median=d[d.source == "MSD-Task07"].dice_tumor.median(),
            voxel_tumor=vox,
            scan_detection=dict(**det, sensitivity=det["TP"] / max(det["TP"] + det["FN"], 1),
                                specificity=det["TN"] / max(det["TN"] + det["FP"], 1),
                                accuracy=(det["TP"] + det["TN"]) / len(d),
                                sensitivity_by_source={k: float(g.tumor_found.mean()) for k, g in d[d.label == 1].groupby("source")}))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))

    v = df[df.split == "val"]
    confusion_matrix_plot(summary["val"]["scan_detection"], OUT / "confusion_detection_val.png",
                          f"Model 1 tumour detection (val, n={len(v)})", ("No tumour", "Tumour"))
    confusion_matrix_plot(summary["val"]["voxel_tumor"], OUT / "voxel_confusion_val.png",
                          "Tumour voxels (validation)", ("Not tumour", "Tumour"))
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for a, key, sub in [(ax[0], "dice_pancreas", v[v.dice_pancreas.notna()]),
                        (ax[1], "dice_tumor", v[v.source == "MSD-Task07"])]:
        sub = sub.sort_values(key)
        a.bar(sub.case_id, sub[key].fillna(0), color="#1f77b4")
        a.axhline(sub[key].mean(), color="k", ls="--", label=f"mean {sub[key].mean():.3f}")
        a.set_ylim(0, 1); a.set_title(f"Validation {key.replace('_', ' ')}"); a.legend()
        a.tick_params(axis="x", rotation=90, labelsize=6)
    fig.tight_layout(); fig.savefig(OUT / "dice_per_case.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
