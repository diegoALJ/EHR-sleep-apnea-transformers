import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(labels, probabilities, threshold=0.5):
    y = np.asarray(labels).astype(int)
    p = np.asarray(probabilities).astype(float)
    pred = (p >= threshold).astype(int)
    return {'f1': float(f1_score(y, pred, zero_division=0)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan'),
            # AP is not numerically identical to trapezoidal PR AUC.
            'average_precision': float(average_precision_score(y, p))}
