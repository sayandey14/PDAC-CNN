"""Convert raw DICOM into harmonised NIfTI volumes.

Every scan from both collections goes through the *same* pipeline so the
classifier cannot tell the datasets apart from trivial differences
(voxel size, orientation, field of view, scanner table):

  1. read DICOM series (SimpleITK)
  2. reorient to LPS
  3. resample to 1.5 mm isotropic (linear for CT, nearest for masks)
  4. crop to the patient's body (removes table + air, equalises field of view)

Outputs in data/processed/:
  <case>_ct.nii.gz        int16 HU
  <case>_pancreas.nii.gz  manual pancreas label     (Pancreas-CT only)
  <case>_stomduo.nii.gz   stomach+duodenum contour  (cancer cases only, used for QC)
  cases.csv               one row per case
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from scipy import ndimage
from skimage.draw import polygon

ROOT = Path(__file__).resolve().parents[1] / "data"
RAW, OUT = ROOT / "raw", ROOT / "processed"
SPACING = (1.5, 1.5, 1.5)


def read_series(d: Path) -> sitk.Image:
    r = sitk.ImageSeriesReader()
    r.SetFileNames(r.GetGDCMSeriesFileNames(str(d)))
    return sitk.Cast(r.Execute(), sitk.sitkInt16)


def rasterize_rtstruct(rt_file: Path, ref: sitk.Image, roi_substr: str) -> sitk.Image:
    """Fill RTSTRUCT contours (patient mm coordinates) into a mask on ref's grid."""
    ds = pydicom.dcmread(rt_file)
    names = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
    mask = np.zeros(sitk.GetArrayFromImage(ref).shape, np.uint8)  # z, y, x
    for roi in ds.ROIContourSequence:
        if roi_substr.lower() not in names[roi.ReferencedROINumber].lower():
            continue
        for c in getattr(roi, "ContourSequence", []):
            pts = np.asarray(c.ContourData, float).reshape(-1, 3)
            idx = np.array([ref.TransformPhysicalPointToContinuousIndex(p.tolist()) for p in pts])
            z = int(round(idx[:, 2].mean()))
            if 0 <= z < mask.shape[0]:
                rr, cc = polygon(idx[:, 1], idx[:, 0], mask.shape[1:])
                mask[z, rr, cc] ^= 1  # XOR handles holes / multiple contours per slice
    out = sitk.GetImageFromArray(mask)
    out.CopyInformation(ref)
    return out


def resample(img, spacing, interp, default):
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), spacing)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), spacing,
                         img.GetDirection(), default, img.GetPixelID())


def body_bbox(ct: np.ndarray):
    body = ndimage.binary_opening(ct > -500, iterations=2)
    lab, n = ndimage.label(body)
    if n:
        body = lab == (np.argmax(ndimage.sum(body, lab, range(1, n + 1))) + 1)
    zz, yy, xx = np.where(body)
    return tuple(slice(a.min(), a.max() + 1) for a in (zz, yy, xx))


def process(case):
    cid = case["case_id"]
    if (OUT / f"{cid}_ct.nii.gz").exists():
        return case
    ct = read_series(case["ct_dir"])
    masks = {}
    if case["label"] == 0:
        lab = sitk.ReadImage(str(case["mask_path"]))
        lab.CopyInformation(ct)  # identical grid, verified in QC below
        masks["pancreas"] = sitk.Cast(lab > 0, sitk.sitkUInt8)
    elif case.get("rt_path"):
        masks["stomduo"] = rasterize_rtstruct(case["rt_path"], ct, "stomach")

    ct = resample(sitk.DICOMOrient(ct, "LPS"), SPACING, sitk.sitkLinear, -1024)
    masks = {k: resample(sitk.DICOMOrient(m, "LPS"), SPACING, sitk.sitkNearestNeighbor, 0) for k, m in masks.items()}

    arr = sitk.GetArrayFromImage(ct)
    bb = body_bbox(arr)
    def save(a, name, ref):
        o = sitk.GetImageFromArray(a[bb])
        o.SetSpacing(ref.GetSpacing()); o.SetDirection(ref.GetDirection())
        o.SetOrigin(ref.TransformIndexToPhysicalPoint([int(bb[2].start), int(bb[1].start), int(bb[0].start)]))
        sitk.WriteImage(o, str(OUT / f"{cid}_{name}.nii.gz"), useCompression=True)
    save(arr, "ct", ct)
    for k, m in masks.items():
        save(sitk.GetArrayFromImage(m), k, ct)
        vals = arr[sitk.GetArrayFromImage(m) > 0]
        case[f"{k}_voxels"], case[f"{k}_meanHU"], case[f"{k}_stdHU"] = len(vals), vals.mean(), vals.std()
    case["shape_zyx"] = str(arr[bb].shape)
    return case


def collect_cases():
    cases = []
    for d in sorted((RAW / "Pancreas-CT").glob("*/*/series.json")):
        s = json.loads(d.read_text())
        n = s["PatientID"].split("_")[1]
        lab = next((RAW / "Pancreas-CT-labels").rglob(f"label{n}.nii.gz"), None)
        if lab is None:
            print(f"[skip] {s['PatientID']}: no label")
            continue
        cases.append(dict(case_id=f"N{n}", patient=s["PatientID"], label=0, source="Pancreas-CT",
                          ct_dir=d.parent, mask_path=lab, scanner=s.get("ManufacturerModelName")))
    by_pid = {}
    for d in sorted((RAW / "Pancreatic-CT-CBCT-SEG").glob("*/*/series.json")):
        s = json.loads(d.read_text())
        by_pid.setdefault(s["PatientID"], []).append((s, d.parent))
    for pid, ss in sorted(by_pid.items()):
        ct = [(s, d) for s, d in ss if s["Modality"] == "CT"]
        if len(ct) != 1:
            print(f"[skip] {pid}: {len(ct)} CTs")
            continue
        ct_uid = ct[0][0]["SeriesInstanceUID"]
        rt = None
        for s, d in ss:
            if s["Modality"] != "RTSTRUCT":
                continue
            f = next(d.glob("*.dcm"))
            ref = pydicom.dcmread(f, stop_before_pixels=True).ReferencedFrameOfReferenceSequence[0] \
                .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
            if ref == ct_uid:
                rt = f
        cases.append(dict(case_id=f"C{pid.split('_')[-1]}", patient=pid, label=1, source="Pancreatic-CT-CBCT-SEG",
                          ct_dir=ct[0][1], rt_path=rt, scanner=ct[0][0].get("ManufacturerModelName")))
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = collect_cases()
    print(f"{sum(c['label'] == 0 for c in cases)} normal, {sum(c['label'] == 1 for c in cases)} cancer")
    with ProcessPoolExecutor(3) as ex:
        done = list(ex.map(process, cases))
    df = pd.DataFrame(done)
    old = OUT / "cases.csv"
    if old.exists():  # keep QC stats from earlier runs for cases skipped this time
        prev = pd.read_csv(old).set_index("case_id")
        df = df.set_index("case_id").combine_first(prev).reset_index()
    df.to_csv(old, index=False)
    if "pancreas_stdHU" in df:
        bad = df[(df.label == 0) & (df.pancreas_stdHU > 60)]
        print("label-alignment QC (pancreas HU std > 60):", list(bad.case_id) or "all OK")


if __name__ == "__main__":
    main()
