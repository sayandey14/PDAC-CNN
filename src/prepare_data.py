"""Stream the TCIA data: download one patient -> convert -> delete the DICOM.

Raw DICOM never piles up on disk. At most a few hundred MB exist at any moment
(--workers patients in flight), and the final dataset is ~1 GB of compact
NIfTI instead of ~12 GB of DICOM (or ~25 GB for the full collections).

  Pancreas-CT             -> 80 healthy pancreases (NIH kidney donors) + manual pancreas labels
  Pancreatic-CT-CBCT-SEG  -> 40 locally-advanced pancreatic cancer patients (MSKCC):
                             only the diagnostic planning CT + its RTSTRUCT. The cone-beam
                             CTs are skipped: they're non-contrast, low quality, and repeat
                             scans of the same patients (which would leak patients
                             across train/val).

Every scan goes through the *same* harmonisation so the two datasets can't be
told apart from trivial differences:
  reorient to LPS -> resample to 1.5 mm isotropic -> crop to the body
  (removes the scanner table and air, equalises field of view)

Outputs in data/processed/:
  <case>_ct.nii.gz        int16 HU
  <case>_pancreas.nii.gz  manual pancreas label     (healthy only)
  <case>_stomduo.nii.gz   stomach+duodenum contour  (cancer only, used for QC)
  cases.csv               one row per case

Re-running resumes: finished cases are skipped.
Usage: python src/prepare_data.py [--workers 3]
"""
import argparse
import io
import shutil
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import requests
import SimpleITK as sitk
from scipy import ndimage
from skimage.draw import polygon
from tqdm import tqdm

from common import PROC, ROOT

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
LABELS_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_pancreas_labels-02-05-2017-1.zip"
PLANNING_SCANNERS = {"Brilliance Big Bore", "Discovery ST"}
SPACING = (1.5, 1.5, 1.5)
TMP = ROOT / "data" / "tmp"


# ---------------------------------------------------------------- conversion
def read_series(d) -> sitk.Image:
    r = sitk.ImageSeriesReader()
    r.SetFileNames(r.GetGDCMSeriesFileNames(str(d)))
    return sitk.Cast(r.Execute(), sitk.sitkInt16)


def rasterize_rtstruct(rt_file, ref: sitk.Image, roi_substr: str) -> sitk.Image:
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


def resample(img, interp, default):
    size = [int(round(sz * sp / ns)) for sz, sp, ns in zip(img.GetSize(), img.GetSpacing(), SPACING)]
    return sitk.Resample(img, size, sitk.Transform(), interp, img.GetOrigin(), SPACING,
                         img.GetDirection(), default, img.GetPixelID())


def body_bbox(ct: np.ndarray):
    body = ndimage.binary_opening(ct > -500, iterations=2)
    lab, n = ndimage.label(body)
    if n:
        body = lab == (np.argmax(ndimage.sum(body, lab, range(1, n + 1))) + 1)
    zz, yy, xx = np.where(body)
    return tuple(slice(int(a.min()), int(a.max()) + 1) for a in (zz, yy, xx))


def harmonise(ct: sitk.Image, masks=None):
    """Reorient, resample to 1.5 mm, crop to body. Returns (ct, {name: mask}) as sitk images.
    Also used by predict.py so new scans get exactly the training preprocessing."""
    masks = masks or {}
    ct = resample(sitk.DICOMOrient(ct, "LPS"), sitk.sitkLinear, -1024)
    masks = {k: resample(sitk.DICOMOrient(m, "LPS"), sitk.sitkNearestNeighbor, 0) for k, m in masks.items()}
    bb = body_bbox(sitk.GetArrayFromImage(ct))
    lo, hi = [s.start for s in bb[::-1]], [s.stop for s in bb[::-1]]  # x, y, z
    crop = lambda im: sitk.RegionOfInterest(im, [h - l for l, h in zip(lo, hi)], lo)
    return crop(ct), {k: crop(m) for k, m in masks.items()}


# ---------------------------------------------------------------- streaming
def api_series(collection):
    r = requests.get(f"{API}/getSeries", params={"Collection": collection}, timeout=120)
    r.raise_for_status()
    return r.json()


def fetch_series(uid, dest: Path):
    for attempt in range(3):
        try:
            r = requests.get(f"{API}/getImage", params={"SeriesInstanceUID": uid}, timeout=900)
            r.raise_for_status()
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(dest)
            return dest
        except Exception:
            if attempt == 2:
                raise


def plan_cases(labels_dir: Path):
    cases = []
    for s in api_series("Pancreas-CT"):
        n = s["PatientID"].split("_")[1]
        lab = next(labels_dir.rglob(f"label{n}.nii.gz"), None)
        if s["Modality"] == "CT" and lab is not None:
            cases.append(dict(case_id=f"N{n}", patient=s["PatientID"], label=0, source="Pancreas-CT",
                              ct_uid=s["SeriesInstanceUID"], mask_path=str(lab)))
    by_pid = {}
    for s in api_series("Pancreatic-CT-CBCT-SEG"):
        by_pid.setdefault(s["PatientID"], []).append(s)
    for pid, ss in sorted(by_pid.items()):
        plan = sorted([s for s in ss if s["Modality"] == "CT" and s.get("ManufacturerModelName") in PLANNING_SCANNERS],
                      key=lambda s: -s["ImageCount"])
        if not plan:
            print(f"[skip] {pid}: no planning CT")
            continue
        rts = [s["SeriesInstanceUID"] for s in ss if s["Modality"] == "RTSTRUCT" and "PC" in (s.get("SeriesDescription") or "")]
        cases.append(dict(case_id=f"C{pid.split('_')[-1]}", patient=pid, label=1, source="Pancreatic-CT-CBCT-SEG",
                          ct_uid=plan[0]["SeriesInstanceUID"], rt_uids=rts, scanner=plan[0].get("ManufacturerModelName")))
    return sorted(cases, key=lambda c: c["case_id"])


def process_case(case):
    cid = case["case_id"]
    work = Path(tempfile.mkdtemp(prefix=cid + "_", dir=TMP))
    try:
        ct = read_series(fetch_series(case["ct_uid"], work / "ct"))
        masks = {}
        if case["label"] == 0:
            lab = sitk.ReadImage(case["mask_path"])
            lab.CopyInformation(ct)  # same grid as the DICOM; alignment QC'd via HU stats below
            masks["pancreas"] = sitk.Cast(lab > 0, sitk.sitkUInt8)
        else:
            for i, uid in enumerate(case["rt_uids"]):
                f = next(fetch_series(uid, work / f"rt{i}").rglob("*.dcm"))
                ref = pydicom.dcmread(f, stop_before_pixels=True).ReferencedFrameOfReferenceSequence[0] \
                    .RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID
                if ref == case["ct_uid"]:
                    masks["stomduo"] = rasterize_rtstruct(f, ct, "stomach")
        case["orig_spacing"] = str(tuple(round(s, 3) for s in ct.GetSpacing()))
        ct, masks = harmonise(ct, masks)
        sitk.WriteImage(ct, str(PROC / f"{cid}_ct.nii.gz"), useCompression=True)
        arr = sitk.GetArrayFromImage(ct)
        for k, m in masks.items():
            sitk.WriteImage(m, str(PROC / f"{cid}_{k}.nii.gz"), useCompression=True)
            vals = arr[sitk.GetArrayFromImage(m) > 0]
            case[f"{k}_voxels"] = len(vals)
            case[f"{k}_meanHU"] = float(vals.mean()) if len(vals) else np.nan
            case[f"{k}_stdHU"] = float(vals.std()) if len(vals) else np.nan
        case["shape_zyx"] = str(arr.shape)
        return case
    finally:
        shutil.rmtree(work, ignore_errors=True)  # raw DICOM is deleted no matter what


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    PROC.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)  # leftovers from an interrupted run
    TMP.mkdir(parents=True)

    labels_dir = ROOT / "data" / "labels"  # 1 MB, kept
    if not labels_dir.exists():
        r = requests.get(LABELS_URL, timeout=300)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(labels_dir)

    csv = PROC / "cases.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame(columns=["case_id"])
    todo = [c for c in plan_cases(labels_dir) if c["case_id"] not in set(done.case_id)]
    print(f"{len(done)} cases already done, {len(todo)} to stream")

    rows, lock = done.to_dict("records"), threading.Lock()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(process_case, c): c["case_id"] for c in todo}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                row = f.result()
            except Exception as e:
                print(f"[error] {futs[f]}: {e}")
                continue
            row = {k: v for k, v in row.items() if k not in ("rt_uids", "mask_path")}
            with lock:  # checkpoint after every case so an interruption loses nothing
                rows.append(row)
                pd.DataFrame(rows).sort_values("case_id").to_csv(csv, index=False)
    shutil.rmtree(TMP, ignore_errors=True)

    df = pd.read_csv(csv)
    print(df.groupby("label").size().rename({0: "healthy", 1: "cancer"}).to_string())
    bad = df[(df.label == 0) & (df.pancreas_stdHU > 60)]
    print("label-alignment QC (healthy pancreas HU std > 60):", list(bad.case_id) or "all OK")
    if "stomduo_voxels" in df:
        print("cancer cases missing stomach/duodenum contour:", list(df[(df.label == 1) & df.stomduo_voxels.isna()].case_id) or "none")


if __name__ == "__main__":
    main()
