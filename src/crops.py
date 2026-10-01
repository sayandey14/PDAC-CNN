"""Cut a fixed-size physical box around the (predicted) pancreas of a scan.

Feeding the classifier the whole scan lets it learn dataset differences (the
cancer CTs cover a different body length, for example) instead of the pancreas
itself. A fixed 1.5 mm box centred on Model 1's predicted pancreas makes every
input the same physical size and anatomy.

The box is cut slightly larger than the network input so training can
random-shift it. Crops are cut on the fly from data/processed (nothing extra
stored on disk).

  region "pancreas"  box centred on the predicted pancreas
  region "control"   same box moved 90 mm toward the patient's right (mostly
                     liver). A CONTROL: if the classifier is just as accurate
                     here, it is detecting scanner / hospital differences,
                     not cancer.
"""
import numpy as np
import SimpleITK as sitk

from common import PROC

CROP = (80, 112, 144)  # z, y, x voxels at 1.5 mm = 120 x 168 x 216 mm (network input is 64 x 96 x 128)
FILL = {"ct": -1024}


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


def box_center(pancreas_mask, region="pancreas"):
    if pancreas_mask.any():
        center = [(v.min() + v.max()) / 2 for v in np.where(pancreas_mask)]  # bbox centre
    else:
        center = [s / 2 for s in pancreas_mask.shape]
    if region == "control":
        center[2] -= 90 / 1.5  # LPS: -x = patient right
    return center


def read(cid, name):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{cid}_{name}.nii.gz")))


def crop_case(cid, region="pancreas", channels=("ct",)):
    """-> float16 array (len(channels), *CROP). channels from: ct, predpancreas, predtumor, tumor."""
    center = box_center(read(cid, "predpancreas") > 0, region)
    out = []
    for ch in channels:
        p = PROC / f"{cid}_{ch}.nii.gz"
        a = read(cid, ch) if p.exists() else np.zeros(1, np.uint8)
        out.append(cut(a, center, CROP, FILL.get(ch, 0)) if a.ndim == 3 else np.zeros(CROP))
    return np.stack(out).astype(np.float16)
