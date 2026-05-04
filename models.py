import torch.nn as nn
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
)


ARCHES = ["resnet50", "efficientnet_b0", "mobilenet_v2"]


def build_model(arch, num_classes=10, pretrained=True):
    arch = arch.lower()

    if arch == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if arch in {"efficientnet_b0", "efficientnet-b0"}:
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if arch in {"mobilenet_v2", "mobilenet-v2"}:
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unknown architecture '{arch}'. Pick one of: {', '.join(ARCHES)}")


def freeze_backbone(model, arch):
    for param in model.parameters():
        param.requires_grad = False

    for param in classifier_parameters(model, arch):
        param.requires_grad = True


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def classifier_parameters(model, arch):
    arch = arch.lower()

    if arch == "resnet50":
        return model.fc.parameters()

    if arch in {"efficientnet_b0", "efficientnet-b0", "mobilenet_v2", "mobilenet-v2"}:
        return model.classifier.parameters()

    raise ValueError(f"Unknown architecture '{arch}'.")


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())

