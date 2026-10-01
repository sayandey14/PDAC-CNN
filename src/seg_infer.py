"""Run Model 1 (pancreas segmentation) on every scan, evaluate it, and save QC.

Predicted masks feed Model 2: the classifier's input box is centred on them.
Both classes are localised the same way (predicted, never manual labels), so
the classifier can't tell them apart by how the box was placed.

Evaluation (outputs/segmentation/):
  metrics.json           Dice + voxel TP/FP/FN/TN, precision/recall, train vs val
  dice_per_case.png      validation Dice per scan
  voxel_confusion.png    voxel-level confusion matrix on validation scans
  qc/<case>.png          overlays (red = prediction, green = manual label or,
                         for cancer scans, the stomach/duodenum contour)

The cancer scans have no pancreas labels, so for them we check that the
prediction sits next to the stomach/duodenum contour (the pancreatic head lies
in the duodenal C-loop).
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
from plots import confusion_matrix_plot
from seg_train import OUT, PATCH, build_model
from split import get_split


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
    x = torch.from_numpy((np.clip(ct_hu, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).float()[None, None].to(dev)
    with torch.no_grad():
        prob = sliding_window_inference(x, PATCH, 4, model, overlap=0.5, mode="gaussian").softmax(1)[0, 1]
    return keep_main_components(prob.cpu().numpy() > 0.5)


def qc_png(cid, ct, pred, ref, path, title):
    zz, yy, xx = np.where(pred if pred.any() else np.ones_like(pred))
    z, y = int(np.median(zz)), int(np.median(yy))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, img, p, g in [(ax[0], ct[z], pred[z], None if ref is None else ref[z]),
                         (ax[1], ct[:, y][::-1], pred[:, y][::-1], None if ref is None else ref[:, y][::-1])]:
        a.imshow(np.clip(img, HU_MIN, HU_MAX), cmap="gray")
        if p.any():
            a.contour(p, [0.5], colors="r", linewidths=1)
        if g is not None and g.any():
            a.contour(g, [0.5], colors="lime", linewidths=1)
        a.axis("off")
    ax[0].set_title(f"{cid} axial — {title}", fontsize=9)
    ax[1].set_title("coronal", fontsize=9)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


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
        pred = segment(model, ct, dev)
        out = sitk.GetImageFromArray(pred.astype(np.uint8))
        out.CopyInformation(img)
        sitk.WriteImage(out, str(PROC / f"{c.case_id}_predpancreas.nii.gz"), useCompression=True)

        row = dict(case_id=c.case_id, label=c.label, split=part[c.case_id], pred_ml=pred.sum() * 1.5 ** 3 / 1000)
        if c.label == 0:
            ref = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{c.case_id}_pancreas.nii.gz"))) > 0
            row.update(TP=int((pred & ref).sum()), FP=int((pred & ~ref).sum()), FN=int((~pred & ref).sum()),
                       TN=int((~pred & ~ref).sum()))
            row["dice"] = 2 * row["TP"] / max(2 * row["TP"] + row["FP"] + row["FN"], 1)
            title = f"{row['split']} Dice {row['dice']:.2f}"
        else:
            ref = None
            if (PROC / f"{c.case_id}_stomduo.nii.gz").exists():
                ref = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{c.case_id}_stomduo.nii.gz"))) > 0
                if ref.any() and pred.any():
                    row["mm_to_stomach_duodenum"] = float(ndimage.distance_transform_edt(~ref, sampling=1.5)[pred].min())
            title = f"cancer, {row.get('mm_to_stomach_duodenum', float('nan')):.0f} mm from stomach/duodenum"
        qc_png(c.case_id, ct, pred, ref, OUT / "qc" / f"{c.case_id}.png", title)
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_case.csv", index=False)

    summary = {}
    for s in ["train", "val"]:
        d = df[(df.label == 0) & (df.split == s)]
        tp, fp, fn, tn = (int(d[k].sum()) for k in ["TP", "FP", "FN", "TN"])
        summary[s] = dict(n_scans=len(d), mean_dice=d.dice.mean(), median_dice=d.dice.median(), min_dice=d.dice.min(),
                          voxel_TP=tp, voxel_FP=fp, voxel_FN=fn, voxel_TN=tn,
                          voxel_precision=tp / max(tp + fp, 1), voxel_recall=tp / max(tp + fn, 1))
    cancer = df[df.label == 1]
    summary["cancer_localisation"] = dict(
        n_scans=len(cancer), empty_predictions=int((cancer.pred_ml == 0).sum()),
        median_mm_to_stomach_duodenum=cancer.get("mm_to_stomach_duodenum", pd.Series(dtype=float)).median(),
        within_15mm=int((cancer.get("mm_to_stomach_duodenum", pd.Series(dtype=float)) <= 15).sum()))
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))

    v = df[(df.label == 0) & (df.split == "val")].sort_values("dice")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(v.case_id, v.dice, color="#1f77b4")
    ax.axhline(v.dice.mean(), color="k", ls="--", label=f"mean {v.dice.mean():.3f}")
    ax.set_ylim(0, 1); ax.set_ylabel("Dice"); ax.set_title("Pancreas segmentation — validation scans"); ax.legend()
    plt.xticks(rotation=60); fig.tight_layout(); fig.savefig(OUT / "dice_per_case.png", dpi=110); plt.close(fig)
    sv = summary["val"]
    confusion_matrix_plot(dict(TP=sv["voxel_TP"], FP=sv["voxel_FP"], FN=sv["voxel_FN"], TN=sv["voxel_TN"]),
                          OUT / "voxel_confusion.png", "Segmentation voxels (validation)", ("Background", "Pancreas"))


if __name__ == "__main__":
    main()
