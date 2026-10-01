"""One patient-level, class-stratified 85/15 train/validation split shared by
BOTH models. A validation patient is never seen by the segmentation model or
the classifier during training, so the validation numbers measure the whole
two-model system on unseen patients.

Stratified by source dataset (healthy / CBCT-SEG cancer / MSD tumour), so
validation holds the same 15% share of each.

Each patient has exactly one scan, so splitting by scan = splitting by patient.
(The legacy code also used cancer-patient cone-beam CTs, i.e. several scans of
the same person, which can leak a patient across train and val.)
"""
import json

from sklearn.model_selection import train_test_split

from common import OUTPUTS, load_cases

VAL_FRACTION = 0.15
SEED = 42


def get_split():
    path = OUTPUTS / "split.json"
    if path.exists():
        return json.loads(path.read_text())
    cases = load_cases().sort_values("case_id")
    tr, va = train_test_split(cases.case_id.tolist(), test_size=VAL_FRACTION, stratify=cases.source, random_state=SEED)
    split = {"train": sorted(tr), "val": sorted(va)}
    OUTPUTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(split, indent=1))
    return split


if __name__ == "__main__":
    s = get_split()
    lab = load_cases().set_index("case_id").label
    src = load_cases().set_index("case_id").source
    for k, ids in s.items():
        print(f"{k}: {len(ids)} scans ({int(lab[ids].sum())} tumour, {int((lab[ids] == 0).sum())} healthy) "
              f"{src[ids].value_counts().to_dict()}")
