"""
Optimization Configurations
Toggle individual optimizations to measure their impact on training throughput.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Toggle individual training optimizations."""
    amp: bool = False                    # Automatic Mixed Precision
    compile: bool = False                # torch.compile (PyTorch 2.0+)
    pin_memory: bool = False             # Pin DataLoader memory
    persistent_workers: bool = False     # Keep DataLoader workers alive
    num_workers: int = 0                 # DataLoader worker count
    prefetch_factor: int = 2             # DataLoader prefetch
    grad_accumulation_steps: int = 1     # Gradient accumulation
    channels_last: bool = False          # NHWC memory format

    @classmethod
    def baseline(cls) -> "OptimizationConfig":
        return cls()

    @classmethod
    def amp_only(cls) -> "OptimizationConfig":
        return cls(amp=True)

    @classmethod
    def compile_only(cls) -> "OptimizationConfig":
        return cls(compile=True)

    @classmethod
    def efficient_dataloader(cls) -> "OptimizationConfig":
        return cls(pin_memory=True, persistent_workers=True, num_workers=4, prefetch_factor=3)

    @classmethod
    def all_combined(cls) -> "OptimizationConfig":
        return cls(
            amp=True, compile=True, pin_memory=True,
            persistent_workers=True, num_workers=4,
            prefetch_factor=3, channels_last=True,
        )


def apply_optimizations(
    model: nn.Module,
    config: OptimizationConfig,
    device: torch.device,
) -> tuple[nn.Module, Optional[torch.GradScaler], bool]:
    """
    Apply selected optimizations to model and return updated components.

    Returns:
        (model, grad_scaler, compile_applied)
    """
    scaler = None
    compile_applied = False

    # Channels last memory format (better for convnets on GPU)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        logger.info("Applied channels_last memory format")

    # Mixed precision scaler
    if config.amp:
        scaler = torch.GradScaler(device.type)
        logger.info("Enabled AMP with GradScaler")

    # torch.compile
    if config.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compile_applied = True
            logger.info("Applied torch.compile (reduce-overhead)")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    return model, scaler, compile_applied
