"""Model registry for benchmark targets."""

import torch.nn as nn
import torchvision.models as models


MODEL_REGISTRY = {
    "resnet18": lambda: models.resnet18(weights=None),
    "resnet50": lambda: models.resnet50(weights=None),
    "vit_b_16": lambda: models.vit_b_16(weights=None),
    "mobilenet_v3": lambda: models.mobilenet_v3_small(weights=None),
    "efficientnet_b0": lambda: models.efficientnet_b0(weights=None),
}


def get_model(name: str) -> nn.Module:
    """Get a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name]()


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
