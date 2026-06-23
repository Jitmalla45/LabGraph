import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from utils.helpers import normalize_label


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def compute_micro_macro_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    micro_precision = _safe_div(tp.sum(), tp.sum() + fp.sum())
    micro_recall = _safe_div(tp.sum(), tp.sum() + fn.sum())
    micro_f1 = _safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    per_class_precision = np.array([_safe_div(t, t + f) for t, f in zip(tp, fp)])
    per_class_recall = np.array([_safe_div(t, t + f) for t, f in zip(tp, fn)])
    per_class_f1 = np.array([
        _safe_div(2 * p * r, p + r)
        for p, r in zip(per_class_precision, per_class_recall)
    ])

    return {
        "micro": {
            "precision": float(micro_precision),
            "recall": float(micro_recall),
            "f1_score": float(micro_f1),
        },
        "macro": {
            "precision": float(per_class_precision.mean()) if len(per_class_precision) else 0.0,
            "recall": float(per_class_recall.mean()) if len(per_class_recall) else 0.0,
            "f1_score": float(per_class_f1.mean()) if len(per_class_f1) else 0.0,
        },
        "per_class": {
            "precision": per_class_precision.tolist(),
            "recall": per_class_recall.tolist(),
            "f1_score": per_class_f1.tolist(),
        },
    }


def evaluate_objects(objects):
    y_true = [normalize_label(obj["gt"]) for obj in objects]
    y_pred = [normalize_label(obj.get("pred", "unknown")) for obj in objects]
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    matches = sum(gt == pred for gt, pred in zip(y_true, y_pred))
    object_accuracy = _safe_div(matches, len(y_true))
    scores = compute_micro_macro_from_confusion(cm)
    metrics = {
        "object_accuracy": object_accuracy,
        "micro": scores["micro"],
        "macro": scores["macro"],
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "labels": labels,
        "confusion_matrix": cm,
        "metrics": metrics,
        "classification_report": report,
    }
