import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from utils import CLASSES, IDX_TO_CLASS


def evaluate(y_true, y_pred, class_names=None):
    if class_names is None:
        class_names = CLASSES

    if isinstance(y_true[0], (int, np.integer)):
        y_true_labels = [IDX_TO_CLASS[y] for y in y_true]
        y_pred_labels = [IDX_TO_CLASS[y] for y in y_pred]
    else:
        y_true_labels = y_true
        y_pred_labels = y_pred

    return {
        "accuracy": float(accuracy_score(y_true_labels, y_pred_labels)),
        "precision_macro": float(precision_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_labels, y_pred_labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_labels, y_pred_labels, labels=class_names).tolist(),
        "classification_report": classification_report(y_true_labels, y_pred_labels, labels=class_names, zero_division=0),
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
