"""Run the full two-model system on a new CT scan.

  CT (DICOM folder or .nii/.nii.gz)
   -> same harmonisation as training (LPS, 1.5 mm, body crop)
   -> Model 1: outline the pancreas and any tumour
   -> cut the pancreas-centred box
   -> Model 2: probability of a pancreatic tumour + Grad-CAM heatmap

Prints the verdict and saves a figure with the tumour outline (Model 1) and
where the classifier looked (Model 2).

Usage: python src/predict.py <dicom_dir | scan.nii.gz> [--tag pancreas] [--out fig.png]
Research code only, not a medical device.
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from cls_train import THRESHOLD, prepare
from common import device
from crops import CROP, box_center, cut
from gradcam import CENTRE, figure, gradcam, load_classifier
from prepare_data import harmonise, read_series
from seg_infer import VOX_ML, load_model, segment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--tag", default="pancreas", help="classifier run under outputs/classification/")
    ap.add_argument("--out", default=None, help="where to save the figure")
    args = ap.parse_args()
    dev = device()

    p = Path(args.scan)
    img = read_series(p) if p.is_dir() else sitk.Cast(sitk.ReadImage(str(p)), sitk.sitkInt16)
    img, _ = harmonise(img)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)

    organ, tumor = segment(load_model(dev), ct, dev)
    if not organ.any():
        print("warning: no pancreas found, using the scan centre")
    center = box_center(organ)
    crops = [cut(a, center, CROP, f) for a, f in [(ct, -1024), (organ.astype(np.float32), 0), (tumor.astype(np.float32), 0)]]

    model = load_classifier(args.tag, dev)
    n_ch = model.features[0].in_channels
    x = torch.from_numpy(np.stack(crops[:2])).float()[None].to(dev)
    prob, cam = gradcam(model, prepare(x, False, 1.0)[:, :n_ch])

    tumor_ml = tumor.sum() * VOX_ML
    verdict = "TUMOUR SUSPECTED" if prob >= THRESHOLD else "no tumour detected"
    print(f"pancreas volume:          {organ.sum() * VOX_ML:.1f} ml")
    print(f"Model 1 tumour outline:   {tumor_ml:.1f} ml" + ("" if tumor_ml else " (none found)"))
    print(f"Model 2 P(tumour) = {prob:.3f}  ->  {verdict}")

    out = Path(args.out or f"prediction_{p.name.split('.')[0]}.png")
    c_ct, c_org, c_tum = (a[CENTRE] for a in crops)
    figure(p.name, int(prob >= THRESHOLD), prob, c_ct, cam, c_org > 0.5, c_tum > 0.5, None, out)
    print(f"figure saved to {out}")


if __name__ == "__main__":
    main()
