"""
Configurable Training Loop
Supports toggling optimizations independently to measure their impact.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .optimizations import OptimizationConfig, apply_optimizations
from .models import get_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics collected during a training run."""
    epoch_times: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    throughput_samples_per_sec: list[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    total_time_sec: float = 0.0

    @property
    def avg_throughput(self) -> float:
        return sum(self.throughput_samples_per_sec) / max(len(self.throughput_samples_per_sec), 1)

    @property
    def avg_epoch_time(self) -> float:
        return sum(self.epoch_times) / max(len(self.epoch_times), 1)

    def summary(self) -> dict:
        return {
            "avg_throughput_img_per_sec": round(self.avg_throughput, 1),
            "avg_epoch_time_sec": round(self.avg_epoch_time, 2),
            "total_time_sec": round(self.total_time_sec, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "final_loss": round(self.losses[-1], 4) if self.losses else None,
        }


class Trainer:
    """
    PyTorch trainer with pluggable optimizations.

    Usage:
        trainer = Trainer(model_name="resnet50", opt_config=OptimizationConfig(amp=True))
        metrics = trainer.train(train_loader, epochs=10)
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        opt_config: Optional[OptimizationConfig] = None,
        device: str = "auto",
        lr: float = 0.001,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.model = get_model(model_name).to(self.device)
        self.opt_config = opt_config or OptimizationConfig()
        self.lr = lr

        # Apply optimizations
        self.model, self.scaler, self.compile_applied = apply_optimizations(
            self.model, self.opt_config, self.device
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        logger.info(
            f"Trainer initialized: model={model_name}, device={self.device}, "
            f"amp={self.opt_config.amp}, compile={self.opt_config.compile}"
        )

    def train(self, train_loader: DataLoader, epochs: int = 10) -> TrainingMetrics:
        """Run training loop and collect metrics."""
        metrics = TrainingMetrics()
        self.model.train()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t_start = time.perf_counter()

        for epoch in range(epochs):
            t_epoch = time.perf_counter()
            running_loss = 0.0
            n_samples = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device, non_blocking=self.opt_config.pin_memory)
                targets = targets.to(self.device, non_blocking=self.opt_config.pin_memory)

                batch_size = inputs.size(0)

                if self.opt_config.amp:
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)

                    self.scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.opt_config.grad_accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()

                    if (batch_idx + 1) % self.opt_config.grad_accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

                running_loss += loss.item() * batch_size
                n_samples += batch_size

            epoch_time = time.perf_counter() - t_epoch
            epoch_loss = running_loss / max(n_samples, 1)
            throughput = n_samples / epoch_time

            metrics.epoch_times.append(epoch_time)
            metrics.losses.append(epoch_loss)
            metrics.throughput_samples_per_sec.append(throughput)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Throughput: {throughput:.0f} img/s | "
                f"Time: {epoch_time:.1f}s"
            )

        metrics.total_time_sec = time.perf_counter() - t_start

        if torch.cuda.is_available():
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / 1e6

        return metrics
