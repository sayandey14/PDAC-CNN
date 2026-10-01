"""Two non-deep-learning reference points, using the same 85/15 split as both models.

1. shortcut check: logistic regression on the scanned body length only
   (z-extent of the scan). This has nothing to do with cancer. If it scores
   highly, the two datasets can be told apart by acquisition alone, and every
   accuracy number in this project must be read with that in mind.
2. pancreas features: logistic regression on simple, interpretable features of
   the predicted pancreas (volume, HU statistics, fraction of low-attenuation
   tissue). Pancreatic cancer typically shows a hypo-attenuating mass, an
   enlarged head and/or an atrophic tail.
"""
import json

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUTPUTS, PROC, load_cases
from plots import summary_metrics
from split import get_split


def pancreas_features(cid):
    ct = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{cid}_ct.nii.gz"))).astype(float)
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(PROC / f"{cid}_predpancreas.nii.gz"))) > 0
    v = ct[m] if m.any() else np.array([0.0])
    zz, yy, xx = np.where(m) if m.any() else (np.zeros(1),) * 3
    return dict(volume_ml=m.sum() * 1.5 ** 3 / 1000, mean_hu=v.mean(), std_hu=v.std(),
                p10_hu=np.percentile(v, 10), p50_hu=np.percentile(v, 50), p90_hu=np.percentile(v, 90),
                frac_below_60hu=(v < 60).mean(), lr_extent_mm=np.ptp(xx) * 1.5, si_extent_mm=np.ptp(zz) * 1.5,
                ap_extent_mm=np.ptp(yy) * 1.5)


def main():
    (OUTPUTS / "classification").mkdir(parents=True, exist_ok=True)
    cases = load_cases().sort_values("case_id").reset_index(drop=True)
    y = cases.label.values
    is_val = cases.case_id.isin(get_split()["val"]).values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))

    z_extent = cases.shape_zyx.map(lambda s: int(s.strip("()").split(",")[0]) * 1.5).values[:, None]
    feats = pd.DataFrame([pancreas_features(c) for c in cases.case_id])
    feats.insert(0, "case_id", cases.case_id)
    feats.insert(1, "label", y)
    feats.to_csv(OUTPUTS / "classification" / "pancreas_features.csv", index=False)

    results = {}
    for name, X in [("shortcut_body_length_only", z_extent),
                    ("pancreas_features_logreg", feats.drop(columns=["case_id", "label"]).values)]:
        clf.fit(X[~is_val], y[~is_val])
        results[name] = summary_metrics(y[is_val], clf.predict_proba(X[is_val])[:, 1])
        print(name, json.dumps(results[name], default=float))
    print(feats.groupby("label").median(numeric_only=True).T.round(2).to_string())
    (OUTPUTS / "classification" / "baselines.json").write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
