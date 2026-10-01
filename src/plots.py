"""Shared figures: learning curves, confusion matrix, ROC / PR curves, probability histogram."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

TRAIN_C, VAL_C = "#1f77b4", "#d62728"


def learning_curves(hist, path, metric_name="Accuracy", title=""):
    """hist: list of dicts with epoch, train_loss, val_loss, train_metric, val_metric (val may be None)."""
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, key, name in [(ax[0], "loss", "Loss"), (ax[1], "metric", metric_name)]:
        a.plot(ep, [h[f"train_{key}"] for h in hist], color=TRAIN_C, label="train")
        v = [(e, h[f"val_{key}"]) for e, h in zip(ep, hist) if h.get(f"val_{key}") is not None]
        if v:
            a.plot(*zip(*v), color=VAL_C, marker="o" if len(v) < 40 else None, ms=3, label="validation")
        a.set_xlabel("epoch"); a.set_ylabel(name); a.set_title(f"{title} {name.lower()}"); a.grid(alpha=.3); a.legend()
    if metric_name in ("Accuracy", "Dice"):
        ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def confusion_counts(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return dict(TP=int((y & pred).sum()), FP=int((~y & pred).sum()), FN=int((y & ~pred).sum()), TN=int((~y & ~pred).sum()))


def confusion_matrix_plot(c, path, title, labels=("Healthy", "Cancer")):
    m = np.array([[c["TN"], c["FP"]], [c["FN"], c["TP"]]])
    names = np.array([["TN", "FP"], ["FN", "TP"]])
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(m, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{names[i, j]}\n{m[i, j]}\n({m[i, j] / max(m[i].sum(), 1):.0%} of row)", ha="center", va="center",
                    color="white" if m[i, j] > m.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], [f"Pred {l}" for l in labels]); ax.set_yticks([0, 1], [f"True {l}" for l in labels])
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def roc_pr_plot(sets, path):
    """sets: {name: (y, prob)}"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for (name, (y, p)), col in zip(sets.items(), [TRAIN_C, VAL_C]):
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=col, label=f"{name} (AUC {auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=col, label=f"{name} (AP {auc(rc, pr):.3f})")
    ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_xlabel("False positive rate (1 - specificity)"); ax[0].set_ylabel("True positive rate (sensitivity)")
    ax[1].set_xlabel("Recall (sensitivity)"); ax[1].set_ylabel("Precision (PPV)")
    for a, t in zip(ax, ["ROC curve", "Precision-recall curve"]):
        a.set_title(t); a.grid(alpha=.3); a.legend(loc="lower right" if t == "ROC curve" else "lower left")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def prob_histogram(y, p, path, title, thr=0.5):
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(p[y == 0], bins, alpha=.6, color="#2ca02c", label="healthy")
    ax.hist(p[y == 1], bins, alpha=.6, color="#d62728", label="cancer")
    ax.axvline(thr, color="k", ls="--", label=f"threshold {thr}")
    ax.set_xlabel("predicted probability of cancer"); ax.set_ylabel("scans"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def summary_metrics(y, p, thr=0.5):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    c = confusion_counts(y, p >= thr)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    div = lambda a, b: a / b if b else float("nan")
    m = dict(n=len(y), **c, accuracy=div(tp + tn, len(y)), sensitivity_recall=div(tp, tp + fn), specificity=div(tn, tn + fp),
             precision_ppv=div(tp, tp + fp), npv=div(tn, tn + fn), f1=div(2 * tp, 2 * tp + fp + fn),
             balanced_accuracy=(div(tp, tp + fn) + div(tn, tn + fp)) / 2)
    m["auc"] = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    return m
