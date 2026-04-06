"""
PyTorch Profiler Integration
Wraps torch.profiler to capture GPU kernel traces, memory timeline,
and operator-level breakdown during training.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from .trainer import Trainer
from .optimizations import OptimizationConfig

logger = logging.getLogger(__name__)


def profile_training(
    model_name: str = "resnet50",
    epochs: int = 3,
    batch_size: int = 64,
    output_dir: str = "profiles/",
    config_name: str = "baseline",
):
    """
    Profile a training run and save trace data.
    
    Outputs:
        - Chrome trace JSON (viewable in chrome://tracing)
        - Operator summary with GPU/CPU time breakdown
        - Memory timeline
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs = {
        "baseline": OptimizationConfig.baseline(),
        "amp": OptimizationConfig.amp_only(),
        "all": OptimizationConfig.all_combined(),
    }
    opt_config = configs.get(config_name, OptimizationConfig.baseline())

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    trainer = Trainer(model_name=model_name, opt_config=opt_config)
    device = trainer.device

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    logger.info(f"Profiling {model_name} with config '{config_name}'...")

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(str(output_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        trainer.model.train()
        for epoch in range(min(epochs, 2)):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if batch_idx >= 10:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.autocast(device_type=device.type, enabled=opt_config.amp):
                    outputs = trainer.model(inputs)
                    loss = trainer.criterion(outputs, targets)

                loss.backward()
                trainer.optimizer.step()
                trainer.optimizer.zero_grad()

                prof.step()

    # Save operator summary
    summary = prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total")
    summary_path = output_path / f"{config_name}_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info(f"Profile saved to {output_path}/")
    logger.info(f"Operator summary:\n{summary}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Profile training run")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="profiles/")
    parser.add_argument("--config", default="baseline", choices=["baseline", "amp", "all"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    profile_training(args.model, args.epochs, args.batch_size, args.output, args.config)


if __name__ == "__main__":
    main()
