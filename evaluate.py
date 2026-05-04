import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from dataset import build_dataloaders
from models import build_model, count_parameters


# =====================
# METRICS
# =====================
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def precision_recall_f1(cm):
    rows = []
    f1s = []

    for idx in range(cm.shape[0]):
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        support = cm[idx, :].sum()

        rows.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support),
            }
        )
        f1s.append(f1)

    return rows, float(np.mean(f1s))


def class_to_risk_ids(class_names):
    return [config.CLASS_TO_RISK[name] for name in class_names]


def save_confusion_csv(path, cm, labels):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row.tolist())


def evaluate_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []
    pin = torch.cuda.is_available()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="evaluate"):
            images = images.to(config.DEVICE, non_blocking=pin)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    return np.array(y_true), np.array(y_pred)


def measure_latency(model, loader, warmup=10, repeats=100):
    model.eval()
    pin = torch.cuda.is_available()
    image, _ = next(iter(loader))
    image = image[:1].to(config.DEVICE, non_blocking=pin)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / repeats


# =====================
# MAIN
# =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)

    arch = checkpoint["arch"]
    class_names = checkpoint.get("class_names", [f"c{i}" for i in range(config.NUM_CLASSES)])

    _, val_loader, _, _, split = build_dataloaders(batch_size=config.BATCH_SIZE)
    model = build_model(arch, num_classes=len(class_names), pretrained=False).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_pred = evaluate_model(model, val_loader)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    class_rows, macro_f1 = precision_recall_f1(cm)
    accuracy = float((y_true == y_pred).mean())

    risk_lookup = np.array(class_to_risk_ids(class_names))
    risk_true = risk_lookup[y_true]
    risk_pred = risk_lookup[y_pred]
    risk_cm = confusion_matrix(risk_true, risk_pred, len(config.RISK_TIER_NAMES))
    _, risk_macro_f1 = precision_recall_f1(risk_cm)
    risk_accuracy = float((risk_true == risk_pred).mean())

    critical_mask = risk_true == config.CLASS_TO_RISK["c1"]
    critical_to_low_medium = critical_mask & (risk_pred <= config.CLASS_TO_RISK["c5"])

    latency_ms = measure_latency(model, val_loader)
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_csv(output_dir / "confusion_matrix_10class.csv", cm, class_names)
    save_confusion_csv(
        output_dir / "confusion_matrix_risk.csv",
        risk_cm,
        config.RISK_TIER_NAMES,
    )

    per_class = {
        class_name: row
        for class_name, row in zip(class_names, class_rows)
    }
    metrics = {
        "arch": arch,
        "mode": checkpoint.get("mode"),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "risk_accuracy": risk_accuracy,
        "risk_macro_f1": risk_macro_f1,
        "critical_to_low_or_medium_count": int(critical_to_low_medium.sum()),
        "critical_count": int(critical_mask.sum()),
        "latency_ms_per_image_batch1": latency_ms,
        "parameter_count": count_parameters(model),
        "model_file_size_mb": model_size_mb,
        "split": split,
        "per_class": per_class,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Risk Accuracy: {risk_accuracy:.4f}")
    print(f"Risk Macro F1: {risk_macro_f1:.4f}")
    print(f"Critical -> Low/Medium: {critical_to_low_medium.sum()} / {critical_mask.sum()}")
    print(f"Latency: {latency_ms:.2f} ms/image")
    print(f"Saved evaluation files to: {output_dir}")


if __name__ == "__main__":
    main()

