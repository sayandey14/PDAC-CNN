"""Download the two TCIA collections used by this project.

  Pancreas-CT             -> 80 healthy pancreases (NIH kidney donors) + NIfTI pancreas labels
  Pancreatic-CT-CBCT-SEG  -> 40 locally-advanced pancreatic cancer patients (MSKCC)

For the cancer collection only the diagnostic-quality *planning CT* (one per
patient) and the RTSTRUCT drawn on it are downloaded. The cone-beam CTs are
skipped: they are non-contrast, low quality, and are repeat scans of the same
patients (using them would leak patients across train/test splits).

Usage:  python src/download.py [--workers 4]
"""
import argparse
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
LABELS_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/TCIA_pancreas_labels-02-05-2017-1.zip"
ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"
PLANNING_SCANNERS = {"Brilliance Big Bore", "Discovery ST"}


def get_series(collection):
    r = requests.get(f"{API}/getSeries", params={"Collection": collection}, timeout=120)
    r.raise_for_status()
    return r.json()


def download_series(series, out_dir: Path):
    dest = out_dir / series["PatientID"] / series["SeriesInstanceUID"]
    done = dest / ".complete"
    if done.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(f"{API}/getImage", params={"SeriesInstanceUID": series["SeriesInstanceUID"]}, timeout=900)
            r.raise_for_status()
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(dest)
            (dest / "series.json").write_text(json.dumps(series, indent=1))
            done.touch()
            return dest
        except Exception:
            if attempt == 2:
                raise


def select_cancer_series(series):
    """One planning CT per patient + the RTSTRUCT(s) referencing planning-CT contours."""
    by_patient = {}
    for s in series:
        by_patient.setdefault(s["PatientID"], []).append(s)
    chosen = []
    for pid, ss in sorted(by_patient.items()):
        plan = [s for s in ss if s["Modality"] == "CT" and s.get("ManufacturerModelName") in PLANNING_SCANNERS]
        if len(plan) != 1:
            print(f"[warn] {pid}: {len(plan)} planning CT candidates, taking the largest")
            plan = sorted(plan, key=lambda s: -s["ImageCount"])[:1]
        rts = [s for s in ss if s["Modality"] == "RTSTRUCT" and "PC" in (s.get("SeriesDescription") or "")]
        chosen += plan + rts
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    labels_dir = ROOT / "Pancreas-CT-labels"
    if not labels_dir.exists():
        print("Downloading Pancreas-CT NIfTI labels")
        r = requests.get(LABELS_URL, timeout=300)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(labels_dir)

    jobs = []
    normal = [s for s in get_series("Pancreas-CT") if s["Modality"] == "CT"]
    jobs += [(s, ROOT / "Pancreas-CT") for s in normal]
    cancer = select_cancer_series(get_series("Pancreatic-CT-CBCT-SEG"))
    jobs += [(s, ROOT / "Pancreatic-CT-CBCT-SEG") for s in cancer]
    total_gb = sum((s.get("FileSize") or 0) for s, _ in jobs) / 1e9
    print(f"{len(normal)} normal CTs, {len(cancer)} cancer series, ~{total_gb:.1f} GB")

    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download_series, s, d): s for s, d in jobs}
        for f in tqdm(as_completed(futs), total=len(futs)):
            try:
                f.result()
            except Exception as e:
                print(f"[error] {futs[f]['SeriesInstanceUID']}: {e}")


if __name__ == "__main__":
    main()
