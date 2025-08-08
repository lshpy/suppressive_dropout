import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def compute_ece(y_true, y_prob, n_bins=15):
    if y_true.ndim == 1:
        num_classes = y_prob.shape[1]
        y_true_oh = np.eye(num_classes)[y_true]
    else:
        y_true_oh = y_true
        num_classes = y_true.shape[1]

    eces = []
    bins = np.linspace(0, 1, n_bins + 1)
    for c in range(num_classes):
        probs = y_prob[:, c]
        labels = y_true_oh[:, c]
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (probs > lo) & (probs <= hi)
            if np.any(m):
                acc = (labels[m] == (probs[m] > 0.5)).mean()
                conf = probs[m].mean()
                ece += np.abs(conf - acc) * m.mean()
        eces.append(ece)
    return float(np.mean(eces))

def multilabel(y):
    return y.ndim == 2 and y.shape[1] > 1

def evaluate_np(labels_np, preds_np, probs_np):
    is_ml = multilabel(labels_np)
    if is_ml:
        acc = (preds_np == labels_np).all(axis=1).mean() * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
        auc = roc_auc_score(labels_np, probs_np)
    else:
        acc = accuracy_score(labels_np, preds_np) * 100
        f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        f1_micro = f1_score(labels_np, preds_np, average='micro', zero_division=0)
        auc = roc_auc_score(labels_np, probs_np, multi_class='ovr')
    ece = compute_ece(labels_np, probs_np)
    return dict(acc=acc, f1_macro=f1_macro, f1_micro=f1_micro, auc=auc, ece=ece)
