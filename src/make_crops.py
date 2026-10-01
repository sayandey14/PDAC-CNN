"""Cut a fixed-size physical box around the (predicted) pancreas of every scan.

Feeding the classifier the whole scan lets it learn dataset differences (the
cancer CTs are breath-hold radiotherapy scans covering a shorter body region,
for example) instead of the pancreas itself. A fixed 1.5 mm box around the
predicted pancreas makes every input the same physical size and anatomy.

Each crop is 2 channels: CT (HU) and predicted pancreas mask. It is cut
slightly larger than the network input so training can random-shift it.

  --region pancreas  (default)  box centred on the predicted pancreas
  --region control              same box moved 90 mm toward the patient's right
                                (mostly liver). A CONTROL: if the classifier is
                                just as accurate here, it is detecting scanner /
                                hospital differences, not cancer.

Output: data/crops/<region>/<case>.npy  (float16, shape 2 x Z x Y x X)
"""
import argparse

import numpy as np
import SimpleITK as sitk

from common import PROC, ROOT, load_cases

CROP = (80, 112, 144)  # z, y, x voxels at 1.5 mm = 120 x 168 x 216 mm (network input is 64 x 96 x 128)


def cut(a, center, size, fill):
    out = np.full(size, fill, a.dtype)
    src, dst = [], []
    for c, s, n in zip(center, size, a.shape):
        lo = int(round(c)) - s // 2
        a0, a1 = max(lo, 0), min(lo + s, n)
        src.append(slice(a0, a1))
        dst.append(slice(a0 - lo, a1 - lo))
    out[tuple(dst)] = a[tuple(src)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", choices=["pancreas", "control"], default="pancreas")
    args = ap.parse_args()
    out_dir = ROOT / "data" / "crops" / args.region
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in load_cases().itertuples():
        ct = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{c.case_id}_ct.nii.gz")))
        pm = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{c.case_id}_predpancreas.nii.gz")))
        if pm.any():
            zz, yy, xx = np.where(pm)
            center = [(v.min() + v.max()) / 2 for v in (zz, yy, xx)]  # bbox centre
        else:
            print(f"[warn] {c.case_id}: empty prediction, using volume centre")
            center = [s / 2 for s in ct.shape]
        if args.region == "control":
            center[2] -= 90 / 1.5  # LPS: -x = patient right
        crop = np.stack([cut(ct, center, CROP, -1024), cut(pm, center, CROP, 0)]).astype(np.float16)
        np.save(out_dir / f"{c.case_id}.npy", crop)
    print(f"wrote {len(list(out_dir.glob('*.npy')))} crops to {out_dir}")


if __name__ == "__main__":
    main()
