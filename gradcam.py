import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from dataset import build_dataloaders
from models import build_model


# =====================
# GRAD-CAM
# =====================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image, target_class=None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        score = output[:, target_class].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / max(1e-8, cam.max())
        return cam, target_class

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def get_target_layer(model, arch):
    arch = arch.lower()
    if arch == "resnet50":
        return model.layer4[-1]
    if arch in {"efficientnet_b0", "efficientnet-b0"}:
        return model.features[-1]
    if arch in {"mobilenet_v2", "mobilenet-v2"}:
        return model.features[-1]
    raise ValueError(f"Unknown architecture: {arch}")


def tensor_to_image(tensor):
    image = tensor.detach().cpu().clone()
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    image = image * std + mean
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(image)


def overlay_heatmap(base_image, cam, alpha=0.4):
    base = base_image.convert("RGB")
    heat = (cam * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat, mode="L").resize(base.size)

    base_rgba = base.convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 0, 0, 0))
    red.putalpha(heat_img.point(lambda value: int(value * alpha)))
    return Image.alpha_composite(base_rgba, red).convert("RGB")


def pick_standard_images(dataset, per_class=3):
    picked = defaultdict(list)
    selected = []

    for dataset_idx in range(len(dataset)):
        image, label = dataset[dataset_idx]
        if len(picked[label]) < per_class:
            picked[label].append(dataset_idx)
            selected.append((image, label, dataset_idx))

        if len(selected) >= per_class * config.NUM_CLASSES:
            break

    return selected


# =====================
# MAIN
# =====================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--per-class", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    arch = checkpoint["arch"]
    class_names = checkpoint.get("class_names", [f"c{i}" for i in range(config.NUM_CLASSES)])

    _, val_loader, _, _, _ = build_dataloaders(batch_size=1)
    val_dataset = val_loader.dataset

    model = build_model(arch, num_classes=len(class_names), pretrained=False).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gradcam = GradCAM(model, get_target_layer(model, arch))
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = pick_standard_images(val_dataset, per_class=args.per_class)
    if len(selected) < args.per_class * len(class_names):
        print("Warning: validation split did not have enough images for every class.")

    for image, label, dataset_idx in selected:
        image_batch = image.unsqueeze(0).to(config.DEVICE)
        cam, predicted = gradcam(image_batch)

        base = tensor_to_image(image)
        overlay = overlay_heatmap(base, cam)

        true_name = class_names[label]
        pred_name = class_names[predicted]
        filename = f"{dataset_idx:04d}_true-{true_name}_pred-{pred_name}.jpg"
        overlay.save(output_dir / filename, quality=95)

    gradcam.close()
    print(f"Saved {len(selected)} Grad-CAM images to: {output_dir}")


if __name__ == "__main__":
    main()

