"""Stream the Medical Segmentation Decathlon pancreas set (Task07_Pancreas).

281 portal-venous CTs (Memorial Sloan Kettering) with expert outlines of the
pancreas AND the tumour (PDAC, IPMN or neuroendocrine tumour). These give the
segmentation model tumour labels to learn from, and give the classifier 281
more tumour scans (vs 40 in Pancreatic-CT-CBCT-SEG).

The 12.3 GB .tar is read as a *stream*: each file is pulled out of the
download, converted with the same harmonisation as every other scan, and
discarded. The archive is never written to disk. The 139 unlabelled test
scans in the archive are skipped.

Outputs in data/processed/:
  M<nnn>_ct.nii.gz, M<nnn>_pancreas.nii.gz (organ incl. tumour), M<nnn>_tumor.nii.gz
  and rows appended to cases.csv (label = 1, source = MSD-Task07)

Usage: python src/prepare_msd.py
"""
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import SimpleITK as sitk
from tqdm import tqdm

from common import PROC, ROOT
from prepare_data import harmonise

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"
TMP = ROOT / "data" / "tmp_msd"


def read_member(tar, member):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", dir=TMP, delete=False) as f:
        shutil.copyfileobj(tar.extractfile(member), f)
    img = sitk.ReadImage(f.name)
    Path(f.name).unlink()
    return img


def save_label(cid, lab: sitk.Image):
    """Resample a label (original grid) onto the already-harmonised CT grid."""
    ref = sitk.ReadImage(str(PROC / f"{cid}_ct.nii.gz"))
    lab = sitk.Resample(sitk.DICOMOrient(sitk.Cast(lab, sitk.sitkUInt8), "LPS"), ref, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(lab)
    ct = sitk.GetArrayFromImage(ref)
    stats = {}
    for name, m in [("pancreas", arr >= 1), ("tumor", arr == 2)]:
        out = sitk.GetImageFromArray(m.astype(np.uint8))
        out.CopyInformation(ref)
        sitk.WriteImage(out, str(PROC / f"{cid}_{name}.nii.gz"), useCompression=True)
        v = ct[m]
        stats.update({f"{name}_voxels": int(m.sum()), f"{name}_meanHU": float(v.mean()) if len(v) else np.nan,
                      f"{name}_stdHU": float(v.std()) if len(v) else np.nan})
    return stats


def main():
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)
    csv = PROC / "cases.csv"
    df = pd.read_csv(csv)
    done = set(df.case_id)
    rows = {}            # case_id -> row being built
    pending_labels = {}  # labels that arrive before their image

    r = requests.get(URL, stream=True, timeout=600)
    r.raise_for_status()
    r.raw.decode_content = True
    bar = tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True)
    raw_read = r.raw.read
    def counted(n=-1):
        b = raw_read(n)
        bar.update(len(b))
        return b
    r.raw.read = counted

    with tarfile.open(fileobj=r.raw, mode="r|") as tar:
        for m in tar:
            name = Path(m.name).name
            hit = re.fullmatch(r"pancreas_(\d+)\.nii\.gz", name)
            kind = Path(m.name).parent.name
            if not m.isfile() or not hit or kind not in ("imagesTr", "labelsTr"):
                continue  # test images, macOS '._' files, json...
            cid = f"M{int(hit.group(1)):03d}"
            if cid in done:
                continue
            if kind == "imagesTr":
                img = sitk.Cast(read_member(tar, m), sitk.sitkInt16)
                spacing = str(tuple(round(s, 3) for s in img.GetSpacing()))
                ct, _ = harmonise(img)
                sitk.WriteImage(ct, str(PROC / f"{cid}_ct.nii.gz"), useCompression=True)
                rows[cid] = dict(case_id=cid, patient=f"pancreas_{hit.group(1)}", label=1, source="MSD-Task07",
                                 orig_spacing=spacing, shape_zyx=str(sitk.GetArrayFromImage(ct).shape))
                if cid in pending_labels:
                    rows[cid].update(save_label(cid, pending_labels.pop(cid)))
            else:
                lab = read_member(tar, m)
                if cid in rows:
                    rows[cid].update(save_label(cid, lab))
                else:
                    pending_labels[cid] = lab
            finished = [c for c, row in rows.items() if "tumor_voxels" in row]
            if finished:  # checkpoint completed cases
                df = pd.concat([df, pd.DataFrame([rows.pop(c) for c in finished])], ignore_index=True)
                df.sort_values("case_id").to_csv(csv, index=False)
                done.update(finished)
    bar.close()
    shutil.rmtree(TMP, ignore_errors=True)
    if rows or pending_labels:
        print(f"[warn] incomplete (image without label or vice versa): {sorted(rows) + sorted(pending_labels)}")
    msd = df[df.source == "MSD-Task07"]
    print(f"MSD cases: {len(msd)}; with visible tumour: {(msd.tumor_voxels > 0).sum()}; "
          f"median tumour volume {msd.tumor_voxels.median() * 1.5 ** 3 / 1000:.1f} ml")


if __name__ == "__main__":
    main()
