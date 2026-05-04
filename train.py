import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import config
from dataset import build_dataloaders
from models import ARCHES, build_model, classifier_parameters, count_parameters
from models import freeze_backbone, unfreeze_all
from plots import plot_training_curves


# =====================
# REPRODUCIBILITY
# =====================
def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_as_dict(args, split):
    return {
        "arch": args.arch,
        "mode": args.mode,
        "seed": config.SEED,
        "batch_size": config.BATCH_SIZE,
        "head_epochs": config.HEAD_EPOCHS,
        "fine_tune_epochs": config.FINE_TUNE_EPOCHS,
        "total_epochs": config.TOTAL_EPOCHS,
        "lr_head": config.LR_HEAD,
        "lr_full": config.LR_FULL,
        "weight_decay": config.WEIGHT_DECAY,
        "train_dir": str(config.TRAIN_DIR),
        "driver_csv": str(config.DRIVER_CSV),
        "split_json": str(config.SPLIT_JSON),
        "split": split,
    }


# =====================
# TRAIN / VAL HELPERS
# =====================
def run_one_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    correct = 0
    total = 0
    pin = torch.cuda.is_available()
    desc = "train" if is_train else "val"

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(config.DEVICE, non_blocking=pin)
        labels = labels.to(config.DEVICE, non_blocking=pin)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, total), correct / max(1, total)


def make_optimizer(model, args, phase):
    if phase == "head":
        params = classifier_parameters(model, args.arch)
        lr = config.LR_HEAD
    else:
        params = model.parameters()
        lr = config.LR_FULL

    return optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, args, class_names):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "arch": args.arch,
        "mode": args.mode,
        "class_names": class_names,
        "num_classes": len(class_names),
    }
    torch.save(checkpoint, path)


def write_log_header(log_path):
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "phase",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "epoch_seconds",
                "lr",
                "peak_gpu_memory_mb",
            ]
        )


def append_log(log_path, row):
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train_phase(model, train_loader, val_loader, criterion, args, run_dir, class_names,
                start_epoch, num_epochs, phase, best_val_acc):
    if phase == "head":
        freeze_backbone(model, args.arch)
    else:
        unfreeze_all(model)

    optimizer = make_optimizer(model, args, phase)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    log_path = run_dir / "training_log.csv"

    for local_epoch in range(num_epochs):
        epoch = start_epoch + local_epoch
        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion)

        epoch_seconds = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        peak_memory_mb = 0.0
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print(f"Epoch [{epoch}/{config.TOTAL_EPOCHS}] ({phase})")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Time: {epoch_seconds:.1f}s | LR: {lr:.6f} | Peak GPU MB: {peak_memory_mb:.1f}")

        append_log(
            log_path,
            [
                epoch,
                phase,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{epoch_seconds:.2f}",
                f"{lr:.8f}",
                f"{peak_memory_mb:.2f}",
            ],
        )

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                run_dir / "best_checkpoint.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val_acc,
                args,
                class_names,
            )
            print("  saved best_checkpoint.pth")

    return best_val_acc


# =====================
# MAIN
# =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=ARCHES, default="resnet50")
    parser.add_argument(
        "--mode",
        choices=["full", "head-only"],
        default="full",
        help="full = 3 head epochs then fine-tune, head-only = frozen backbone all epochs",
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed()

    print("DEVICE:", config.DEVICE)
    print("Architecture:", args.arch)
    print("Mode:", args.mode)

    train_loader, val_loader, class_weights, class_names, split = build_dataloaders()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(args.arch, num_classes=len(class_names)).to(config.DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    run_name = args.run_name
    if run_name is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.arch}_{args.mode}_{stamp}"

    run_dir = Path(config.RUNS_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_log_header(run_dir / "training_log.csv")

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config_as_dict(args, split), f, indent=2)

    best_val_acc = 0.0

    if args.mode == "head-only":
        best_val_acc = train_phase(
            model,
            train_loader,
            val_loader,
            criterion,
            args,
            run_dir,
            class_names,
            start_epoch=1,
            num_epochs=config.TOTAL_EPOCHS,
            phase="head",
            best_val_acc=best_val_acc,
        )
    else:
        best_val_acc = train_phase(
            model,
            train_loader,
            val_loader,
            criterion,
            args,
            run_dir,
            class_names,
            start_epoch=1,
            num_epochs=config.HEAD_EPOCHS,
            phase="head",
            best_val_acc=best_val_acc,
        )
        best_val_acc = train_phase(
            model,
            train_loader,
            val_loader,
            criterion,
            args,
            run_dir,
            class_names,
            start_epoch=config.HEAD_EPOCHS + 1,
            num_epochs=config.FINE_TUNE_EPOCHS,
            phase="fine-tune",
            best_val_acc=best_val_acc,
        )

    final_path = run_dir / "final_checkpoint.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "arch": args.arch,
            "mode": args.mode,
            "class_names": class_names,
            "best_val_acc": best_val_acc,
        },
        final_path,
    )

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    plot_training_curves(run_dir / "training_log.csv", run_dir / "figures")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()

