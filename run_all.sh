#!/usr/bin/env bash
# Full two-model pipeline. ~4-5 hours on an Apple M3 (8 GB), mostly training.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY src/prepare_data.py                       # stream TCIA: download -> convert -> delete DICOM (~1.5 GB kept)
$PY src/split.py                              # shared 85/15 patient split
$PY src/seg_train.py                          # Model 1: pancreas segmentation (3D U-Net)
$PY src/seg_infer.py                          # predicted pancreas for every scan + seg metrics/QC
$PY src/make_crops.py --region pancreas
$PY src/make_crops.py --region control
$PY src/baselines.py                          # shortcut check + simple feature model
$PY src/cls_train.py --region pancreas        # Model 2: cancer classifier (main result)
$PY src/cls_train.py --region control --no_mask   # control: box over the liver, not the pancreas
