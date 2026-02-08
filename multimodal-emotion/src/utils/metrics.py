from typing import Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred, average_labels=None) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    # per-class F1
    unique = np.unique(y_true) if average_labels is None else average_labels
    f1s = f1_score(y_true, y_pred, average=None, labels=unique, zero_division=0)
    metrics['per_class_f1'] = {int(lbl): float(f1) for lbl, f1 in zip(unique, f1s)}
    return metrics
