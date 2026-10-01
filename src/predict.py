"""Run the full two-model system on a new CT scan.

  CT (DICOM folder or .nii/.nii.gz)
   -> same harmonisation as training (LPS, 1.5 mm, body crop)
   -> Model 1: segment the pancreas
   -> cut the pancreas-centred box
   -> Model 2: probability of pancreatic cancer

Usage: python src/predict.py <dicom_dir | scan.nii.gz> [--model outputs/classification/pancreas]
Research code only, not a medical device.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch

from cls_train import THRESHOLD, Net, predict
from common import HU_MAX, HU_MIN, OUTPUTS, device
from make_crops import CROP, cut
from prepare_data import harmonise, read_series
from seg_infer import load_model, segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--model", default=str(OUTPUTS / "classification" / "pancreas"))
    ap.add_argument("--out", default=None, help="where to save the overlay PNG")
    args = ap.parse_args()
    dev = device()

    p = Path(args.scan)
    img = read_series(p) if p.is_dir() else sitk.Cast(sitk.ReadImage(str(p)), sitk.sitkInt16)
    img, _ = harmonise(img)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)

    mask = segment(load_model(dev), ct, dev)
    if mask.any():
        zz, yy, xx = np.where(mask)
        center = [(v.min() + v.max()) / 2 for v in (zz, yy, xx)]
    else:
        print("warning: no pancreas found, using the scan centre")
        center = [s / 2 for s in ct.shape]

    state = torch.load(Path(args.model) / "model.pt", map_location=dev)
    n_ch = state["features.0.weight"].shape[1]
    model = Net(n_ch).to(dev)
    model.load_state_dict(state)
    crop = np.stack([cut(ct, center, CROP, -1024), cut(mask.astype(np.float32), center, CROP, 0)])[:n_ch]
    prob = float(predict(model, torch.from_numpy(crop)[None], n_ch, blur=1.0, dev=dev)[0])

    verdict = "CANCER SUSPECTED" if prob >= THRESHOLD else "no cancer detected"
    print(f"pancreas volume: {mask.sum() * 1.5 ** 3 / 1000:.1f} ml")
    print(f"P(cancer) = {prob:.3f}  ->  {verdict}")

    out = Path(args.out or f"prediction_{p.name.split('.')[0]}.png")
    z = int(center[0])
    plt.figure(figsize=(6, 6))
    plt.imshow(np.clip(ct[z], HU_MIN, HU_MAX), cmap="gray")
    if mask[z].any():
        plt.contour(mask[z], [0.5], colors="r", linewidths=1)
    plt.title(f"P(cancer) = {prob:.2f} — {verdict}")
    plt.axis("off")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    print(f"overlay saved to {out}")


if __name__ == "__main__":
    main()
