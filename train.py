import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATA_DIR = os.environ.get(
    "DATA_DIR",
    r"C:\Users\keert\Downloads\DistractedDrivingDetection\data\train",
)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
EPOCHS = int(os.environ.get("EPOCHS", "12"))
LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
LR_FULL = float(os.environ.get("LR_FULL", "1e-4"))
VAL_FRAC = float(os.environ.get("VAL_FRAC", "0.2"))
SEED = int(os.environ.get("SEED", "42"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# =====================
# TRANSFORMS
# =====================
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# =====================
# DATASET (fixed: separate ImageFolders so val gets val transforms)
# =====================
train_root = datasets.ImageFolder(DATA_DIR, transform=train_transform)
val_root = datasets.ImageFolder(DATA_DIR, transform=val_transform)

assert len(train_root) == len(val_root)
n = len(train_root)
indices = list(range(n))
random.seed(SEED)
random.shuffle(indices)

val_size = int(round(n * VAL_FRAC))
train_size = n - val_size
train_idx = indices[:train_size]
val_idx = indices[train_size:]

train_dataset = Subset(train_root, train_idx)
val_dataset = Subset(val_root, val_idx)

pin = torch.cuda.is_available()
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin,
)

# =====================
# CLASS WEIGHTS (computed on TRAIN split only)
# =====================
train_targets = [train_root.samples[i][1] for i in train_idx]
class_counts = Counter(train_targets)

num_classes = len(train_root.classes)
total_samples = len(train_targets)

class_weights = torch.tensor(
    [total_samples / max(1, class_counts[i]) for i in range(num_classes)],
    dtype=torch.float32,
    device=DEVICE,
)

# =====================
# MODEL (fixed: weights API)
# =====================
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# Freeze backbone initially
for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

# =====================
# LOSS + OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD)

# =====================
# TRAIN / VAL
# =====================
def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=pin)
        labels = labels.to(DEVICE, non_blocking=pin)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def validate(model, loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="val", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=pin)
        labels = labels.to(DEVICE, non_blocking=pin)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, total), correct / max(1, total)


# =====================
# TRAIN LOOP
# =====================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("  saved best_model.pth")

    # Unfreeze after 3 epochs
    if epoch == 2:
        print("Unfreezing backbone...")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LR_FULL)

print(f"Best Validation Accuracy: {best_val_acc:.4f}")