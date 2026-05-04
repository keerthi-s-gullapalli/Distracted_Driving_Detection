import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


# =====================
# TRANSFORMS
# =====================
def get_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def get_val_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


# =====================
# DRIVER SPLIT
# =====================
def _read_driver_csv(csv_path):
    image_to_driver = {}
    drivers = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            driver = row["subject"]
            classname = row["classname"]
            img = row["img"]
            image_to_driver[(classname, img)] = driver
            drivers.add(driver)

    return image_to_driver, sorted(drivers)


def _make_split(drivers, seed=config.SEED):
    rng = random.Random(seed)
    shuffled = list(drivers)
    rng.shuffle(shuffled)

    val_count = config.NUM_VAL_DRIVERS
    if val_count <= 0:
        val_count = max(1, round(len(shuffled) * config.VAL_DRIVER_FRAC))

    val_drivers = sorted(shuffled[:val_count])
    train_drivers = sorted(shuffled[val_count:])
    return train_drivers, val_drivers


def load_or_create_split(csv_path=config.DRIVER_CSV, split_path=config.SPLIT_JSON):
    csv_path = Path(csv_path)
    split_path = Path(split_path)

    image_to_driver, drivers = _read_driver_csv(csv_path)

    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
    else:
        train_drivers, val_drivers = _make_split(drivers)
        split = {
            "seed": config.SEED,
            "train_drivers": train_drivers,
            "val_drivers": val_drivers,
            "num_drivers": len(drivers),
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)

    return split, image_to_driver


def _indices_for_drivers(image_folder, image_to_driver, wanted_drivers):
    wanted_drivers = set(wanted_drivers)
    indices = []
    missing = []

    for idx, (path, _) in enumerate(image_folder.samples):
        path = Path(path)
        key = (path.parent.name, path.name)
        driver = image_to_driver.get(key)

        if driver is None:
            missing.append(str(path))
            continue

        if driver in wanted_drivers:
            indices.append(idx)

    if missing:
        example = missing[0]
        raise ValueError(
            f"{len(missing)} images were not found in driver_imgs_list.csv. "
            f"Example: {example}"
        )

    return indices


def get_class_weights(image_folder, train_indices, device=config.DEVICE):
    train_targets = [image_folder.samples[i][1] for i in train_indices]
    class_counts = Counter(train_targets)
    total = len(train_targets)

    weights = [
        total / max(1, class_counts[class_idx])
        for class_idx in range(len(image_folder.classes))
    ]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_dataloaders(batch_size=config.BATCH_SIZE):
    train_root = datasets.ImageFolder(config.TRAIN_DIR, transform=get_train_transform())
    val_root = datasets.ImageFolder(config.TRAIN_DIR, transform=get_val_transform())

    split, image_to_driver = load_or_create_split()
    train_idx = _indices_for_drivers(train_root, image_to_driver, split["train_drivers"])
    val_idx = _indices_for_drivers(val_root, image_to_driver, split["val_drivers"])

    if not train_idx or not val_idx:
        raise ValueError("The driver split made an empty train or validation set.")

    train_dataset = Subset(train_root, train_idx)
    val_dataset = Subset(val_root, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin,
    )

    class_weights = get_class_weights(train_root, train_idx)
    split["train_images"] = len(train_idx)
    split["val_images"] = len(val_idx)
    split["classes"] = train_root.classes

    return train_loader, val_loader, class_weights, train_root.classes, split

