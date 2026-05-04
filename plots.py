import csv
import importlib
from pathlib import Path

import numpy as np

import config


def _get_pyplot():
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        return importlib.import_module("matplotlib.pyplot")
    except ImportError:
        print("matplotlib is not installed, so plots were skipped.")
        return None


# =====================
# TRAINING PLOTS
# =====================
def read_training_log(log_path):
    rows = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def plot_training_curves(log_path, output_dir):
    plt = _get_pyplot()
    if plt is None:
        return

    rows = read_training_log(log_path)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_acc"]) for row in rows]
    val_acc = [float(row["val_acc"]) for row in rows]
    lr = [float(row["lr"]) for row in rows]
    epoch_seconds = [float(row["epoch_seconds"]) for row in rows]
    gpu_memory = [float(row["peak_gpu_memory_mb"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", label="Train loss")
    ax.plot(epochs, val_loss, marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_acc, marker="o", label="Train accuracy")
    ax.plot(epochs, val_acc, marker="o", label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Training and Validation Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, lr, marker="o")
    axes[0].set_title("Learning Rate")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("LR")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, epoch_seconds, marker="o", label="Seconds/epoch")
    if max(gpu_memory) > 0:
        axes[1].plot(epochs, gpu_memory, marker="o", label="Peak GPU MB")
    axes[1].set_title("Training Cost")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "training_cost.png", dpi=200)
    plt.close(fig)


# =====================
# EVALUATION PLOTS
# =====================
def plot_confusion_matrix(cm, labels, title, output_path):
    plt = _get_pyplot()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_per_class_metrics(per_class, output_path):
    plt = _get_pyplot()
    if plt is None:
        return

    labels = list(per_class.keys())
    precision = [per_class[label]["precision"] for label in labels]
    recall = [per_class[label]["recall"] for label in labels]
    f1 = [per_class[label]["f1"] for label in labels]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, and F1")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def average_precision_from_scores(y_true_binary, scores):
    order = np.argsort(-scores)
    y_sorted = y_true_binary[order]
    positives = y_sorted.sum()

    if positives == 0:
        return float("nan"), np.array([0.0]), np.array([1.0])

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives

    recall_with_start = np.concatenate([[0.0], recall])
    precision_with_start = np.concatenate([[1.0], precision])
    ap = np.sum((recall_with_start[1:] - recall_with_start[:-1]) * precision_with_start[1:])

    return float(ap), recall_with_start, precision_with_start


def plot_precision_recall_curves(y_true, y_prob, labels, output_path, title):
    ap_scores = {}
    curves = []

    for class_idx, label in enumerate(labels):
        y_binary = (y_true == class_idx).astype(int)
        ap, recall, precision = average_precision_from_scores(y_binary, y_prob[:, class_idx])
        ap_scores[label] = ap
        curves.append((label, ap, recall, precision))

    plt = _get_pyplot()
    if plt is None:
        return ap_scores

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, ap, recall, precision in curves:
        if not np.isnan(ap):
            ax.plot(recall, precision, label=f"{label} AP={ap:.2f}", linewidth=1.5)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return ap_scores


def risk_probabilities(class_probs, class_names):
    risk_probs = np.zeros((class_probs.shape[0], len(config.RISK_TIER_NAMES)))

    for class_idx, class_name in enumerate(class_names):
        risk_idx = config.CLASS_TO_RISK[class_name]
        risk_probs[:, risk_idx] += class_probs[:, class_idx]

    return risk_probs

