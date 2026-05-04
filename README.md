# Distracted Driving Detection

This project compares pretrained CNN models on the State Farm Distracted Driver Detection dataset. The main task is 10-class image classification, where each image is assigned one of the driver behavior labels `c0` through `c9`. After the model predicts a class, the prediction is also mapped into a simpler risk tier: Low, Medium, High, or Critical.

The project was built for a computer vision class, so the code is kept fairly direct and easy to follow instead of being wrapped in a large framework.

## Project Goal

The goal is to compare different CNN architectures under the same training setup:

- ResNet18
- ResNet50
- EfficientNet-B0
- MobileNet-V2

The original plan focused on ResNet50, EfficientNet-B0, and MobileNet-V2. ResNet18 was also added as a lighter ResNet baseline because it trains faster and is useful when compute time is limited.

Each model is trained in two different ways:

- `head-only`: the pretrained backbone stays frozen and only the final classifier layer is trained.
- `full`: the classifier head is trained first, then the full network is unfrozen and fine-tuned.

This lets us compare both the architecture choice and the value of full fine-tuning.

## Dataset

The code expects the State Farm dataset to be arranged like this:

```text
data/
  driver_imgs_list.csv
  train/
    c0/
    c1/
    c2/
    c3/
    c4/
    c5/
    c6/
    c7/
    c8/
    c9/
```

The `driver_imgs_list.csv` file is important because it tells the code which driver appears in each image.

Instead of randomly splitting images, the project splits by driver ID. This is more realistic because the validation set contains drivers the model did not see during training. Random image splits can make validation accuracy look too high because images from the same driver can appear in both train and validation.

On the first run, the code creates:

```text
split_seed42.json
```

After that, all model runs reuse the same split so the comparison is fair.

## Risk Tiers

The model predicts one of the 10 original classes. Then the code maps that prediction into a risk level:

| Risk tier | Classes |
| --- | --- |
| Low | `c0` normal driving |
| Medium | `c5` radio operation, `c9` talking to passenger |
| High | `c2` phone call right, `c4` phone call left, `c6` drinking, `c8` hair/makeup |
| Critical | `c1` texting right, `c3` texting left, `c7` reaching behind |

This risk score is not a separate trained model. It is a deterministic lookup after classification.

## Files

```text
config.py       # paths, hyperparameters, seed, class/risk settings
dataset.py      # driver-aware split, transforms, DataLoaders, class weights
models.py       # model factory for ResNet18, ResNet50, EfficientNet-B0, MobileNet-V2
train.py        # training loop, checkpointing, logs, training plots
evaluate.py     # validation metrics, confusion matrices, AP curves, risk analysis
gradcam.py      # Grad-CAM visualization generation
plots.py        # helper functions for saving report figures
requirements.txt
.gitignore
```

## Installation

Create a virtual environment if you want:

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the requirements:

```bash
pip install -r requirements.txt
```

If PyTorch does not install correctly for your GPU, install it using the official instructions:

https://pytorch.org/get-started/locally/

## Training

The general training command is:

```bash
python train.py --arch MODEL_NAME --mode TRAINING_MODE
```

Supported model names:

```text
resnet18
resnet50
efficientnet_b0
mobilenet_v2
```

Supported training modes:

```text
head-only
full
```

Example:

```bash
python train.py --arch resnet18 --mode full
```

### Full Fine-Tuning

The default full training setup is:

```text
Epochs 1-3: train classifier head only
Epochs 4-15: unfreeze full model and fine-tune
```

The default values are in `config.py`:

```python
HEAD_EPOCHS = 3
FINE_TUNE_EPOCHS = 12
```

The full fine-tuning command for each model is:

```bash
python train.py --arch resnet18 --mode full
python train.py --arch resnet50 --mode full
python train.py --arch efficientnet_b0 --mode full
python train.py --arch mobilenet_v2 --mode full
```

### Head-Only Ablation

For the head-only experiment:

```bash
python train.py --arch resnet18 --mode head-only
python train.py --arch efficientnet_b0 --mode head-only
python train.py --arch mobilenet_v2 --mode head-only
```

In this mode, the backbone stays frozen for all epochs and only the final classifier layer trains.

### Quick Test Run

Before doing a full run, it is useful to check that the dataset paths and code are working:

```bash
HEAD_EPOCHS=1 FINE_TUNE_EPOCHS=1 python train.py --arch resnet18 --mode full
```

On Windows PowerShell:

```powershell
$env:HEAD_EPOCHS="1"
$env:FINE_TUNE_EPOCHS="1"
python train.py --arch resnet18 --mode full
```

## Training Outputs

Each training run creates a folder under `runs/`:

```text
runs/resnet18_full_YYYYMMDD_HHMMSS/
```

Inside each run folder:

```text
best_checkpoint.pth      # best validation checkpoint
final_checkpoint.pth     # model at the final epoch
training_log.csv         # per-epoch train/validation metrics
run_config.json          # hyperparameters and split information
figures/
  loss_curve.png
  accuracy_curve.png
  training_cost.png
```

The `.pth` files are ignored by git because they are large. The logs, configs, and PNG figures are meant to be pushed and used in the report.

## Evaluation

To evaluate a saved checkpoint:

```bash
python evaluate.py --checkpoint runs/resnet18_full_YYYYMMDD_HHMMSS/best_checkpoint.pth
```

For the runs used in the report, it is usually better to evaluate `best_checkpoint.pth` instead of `final_checkpoint.pth`, because validation accuracy often peaks before the last epoch.

Evaluation uses only the validation split, not the training set and not the full dataset. It reloads the same driver-aware split from `split_seed42.json`.

Evaluation outputs are saved in:

```text
runs/<run_name>/eval/
```

The evaluation folder contains:

```text
metrics.json
confusion_matrix_10class.csv
confusion_matrix_10class.png
confusion_matrix_risk.csv
confusion_matrix_risk.png
per_class_metrics.png
precision_recall_10class.png
precision_recall_risk.png
```

The metrics include:

- overall validation accuracy
- macro F1
- per-class precision, recall, and F1
- 10-class confusion matrix
- risk-tier accuracy
- risk-tier macro F1
- average precision by class
- average precision by risk tier
- critical-to-low/medium misclassification count
- inference latency at batch size 1
- parameter count
- checkpoint file size

To evaluate every best checkpoint in PowerShell:

```powershell
Get-ChildItem runs -Recurse -Filter best_checkpoint.pth | ForEach-Object {
    python evaluate.py --checkpoint $_.FullName
}
```

## Grad-CAM

Grad-CAM is used to visualize which regions of the image the model is focusing on.

Run it like this:

```bash
python gradcam.py --checkpoint runs/resnet18_full_YYYYMMDD_HHMMSS/best_checkpoint.pth
```

By default, it saves 3 images per class from the validation set:

```text
runs/<run_name>/gradcam/
```

To save fewer examples:

```bash
python gradcam.py --checkpoint runs/resnet18_full_YYYYMMDD_HHMMSS/best_checkpoint.pth --per-class 1
```

These images are useful in the report because they show whether the model is looking near the driver, hands, phone, face, or other relevant regions instead of just relying on the car interior or background.

## Results Summary

The following validation accuracies came from the driver-aware validation split:

| Model | Head-only best validation accuracy | Full fine-tuning best validation accuracy |
| --- | ---: | ---: |
| ResNet18 | 52.18% | 90.69% |
| EfficientNet-B0 | 49.05% | 84.88% |
| MobileNet-V2 | 51.27% | 84.90% |

The main result is that full fine-tuning was much better than only training the final classifier head. This makes sense because the validation set contains unseen drivers, so the models need to adapt more than they would for a random image split.

ResNet18 had the best validation accuracy in these runs. MobileNet-V2 and EfficientNet-B0 were lower, but they are smaller models and may still be useful when inference speed or memory is more important.

## Notes

- The code uses ImageNet pretrained weights from `torchvision`.
- Training uses weighted cross-entropy based only on the training split.
- Horizontal flipping is intentionally not used because left/right driver actions are separate classes.
- The validation transform does not use augmentation.
- The project saves plots automatically so the report can include training curves, confusion matrices, precision-recall curves, and Grad-CAM examples.

