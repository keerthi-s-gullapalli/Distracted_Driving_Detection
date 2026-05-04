import os
from pathlib import Path

import torch


# =====================
# PATHS
# =====================
# Expected Kaggle layout:
# data/
#   driver_imgs_list.csv
#   train/c0/*.jpg ... train/c9/*.jpg
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
TRAIN_DIR = Path(os.environ.get("TRAIN_DIR", DATA_ROOT / "train"))
DRIVER_CSV = Path(os.environ.get("DRIVER_CSV", DATA_ROOT / "driver_imgs_list.csv"))
SPLIT_JSON = Path(os.environ.get("SPLIT_JSON", PROJECT_ROOT / "split_seed42.json"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", PROJECT_ROOT / "runs"))


# =====================
# TRAINING SETTINGS
# =====================
SEED = int(os.environ.get("SEED", "42"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

HEAD_EPOCHS = int(os.environ.get("HEAD_EPOCHS", "3"))
FINE_TUNE_EPOCHS = int(os.environ.get("FINE_TUNE_EPOCHS", "12"))
TOTAL_EPOCHS = HEAD_EPOCHS + FINE_TUNE_EPOCHS

LR_HEAD = float(os.environ.get("LR_HEAD", "1e-3"))
LR_FULL = float(os.environ.get("LR_FULL", "1e-4"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "1e-4"))

VAL_DRIVER_FRAC = float(os.environ.get("VAL_DRIVER_FRAC", "0.2"))
NUM_VAL_DRIVERS = int(os.environ.get("NUM_VAL_DRIVERS", "6"))
NUM_CLASSES = 10


# =====================
# IMAGE SETTINGS
# =====================
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================
# RISK TIERS
# =====================
RISK_TIER_NAMES = ["Low", "Medium", "High", "Critical"]
CLASS_TO_RISK = {
    "c0": 0,  # normal driving
    "c5": 1,  # radio
    "c9": 1,  # talking to passenger
    "c2": 2,  # phone call right
    "c4": 2,  # phone call left
    "c6": 2,  # drinking
    "c8": 2,  # hair/makeup
    "c1": 3,  # texting right
    "c3": 3,  # texting left
    "c7": 3,  # reaching behind
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True