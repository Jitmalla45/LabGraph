from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_difference_map(diff, path):
    _ensure_parent(path)
    plt.figure(figsize=(10, 8))
    plt.imshow(diff, cmap="hot")
    plt.title("Difference Map")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, labels, path):
    _ensure_parent(path)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def save_metrics_plot(metrics, path):
    _ensure_parent(path)
    names = ["Precision", "Recall", "F1-Score"]
    micro = [
        metrics["micro"]["precision"],
        metrics["micro"]["recall"],
        metrics["micro"]["f1_score"],
    ]
    macro = [
        metrics["macro"]["precision"],
        metrics["macro"]["recall"],
        metrics["macro"]["f1_score"],
    ]
    x = np.arange(len(names))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, micro, width, label="Micro")
    plt.bar(x + width / 2, macro, width, label="Macro")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1-Score")
    plt.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
