#!/usr/bin/env bash
# Full two-model pipeline. ~6-7 hours on an Apple M3 (8 GB), mostly training.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY src/prepare_data.py                  # stream TCIA (healthy + CBCT-SEG cancer): download -> convert -> delete DICOM
$PY src/prepare_msd.py                   # stream MSD Task07 (281 scans with tumour outlines), archive never stored
$PY src/split.py                         # shared 85/15 patient split
$PY src/seg_train.py                     # Model 1: pancreas + tumour segmentation (3D U-Net)
$PY src/seg_infer.py                     # Model 1 on every scan + seg metrics / QC overlays
$PY src/baselines.py                     # shortcut check + simple feature model
$PY src/cls_train.py --region pancreas   # Model 2: tumour classifier (main result)
$PY src/gradcam.py --tag pancreas        # where Model 2 looked, scored against manual tumour outlines
$PY src/cls_train.py --region control    # control: box over the liver instead of the pancreas
