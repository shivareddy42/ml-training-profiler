"""
Benchmark Suite
Runs training with each optimization config and compares results.

Usage:
    python -m src.benchmark --model resnet50 --dataset cifar10 --epochs 5
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .trainer import Trainer, TrainingMetrics
from .optimizations import OptimizationConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


CONFIGS = {
    "baseline": OptimizationConfig.baseline(),
    "amp": OptimizationConfig.amp_only(),
    "compile": OptimizationConfig.compile_only(),
    "efficient_dataloader": OptimizationConfig.efficient_dataloader(),
    "all_combined": OptimizationConfig.all_combined(),
}


def get_dataloader(
    dataset: str, batch_size: int, config: OptimizationConfig
) -> DataLoader:
    """Create DataLoader with optimization-specific settings."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    if dataset == "cifar10":
        ds = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset == "cifar100":
        ds = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        drop_last=True,
    )


def run_benchmark(
    model_name: str,
    dataset: str,
    batch_size: int,
    epochs: int,
    output_dir: str,
    configs: list[str],
):
    """Run benchmark across selected optimization configs."""
    results = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for config_name in configs:
        if config_name not in CONFIGS:
            logger.warning(f"Unknown config '{config_name}', skipping")
            continue

        opt_config = CONFIGS[config_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {config_name}")
        logger.info(f"{'='*60}")

        loader = get_dataloader(dataset, batch_size, opt_config)
        trainer = Trainer(model_name=model_name, opt_config=opt_config)
        metrics = trainer.train(loader, epochs=epochs)

        results[config_name] = metrics.summary()
        logger.info(f"Results: {metrics.summary()}")

        # Save individual result
        with open(output_path / f"{config_name}.json", "w") as f:
            json.dump(metrics.summary(), f, indent=2)

        # Free GPU memory between runs
        del trainer, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"{'Config':<25} {'Throughput':>12} {'Speedup':>10} {'Memory':>10}")
    print(f"{'='*70}")

    baseline_tp = results.get("baseline", {}).get("avg_throughput_img_per_sec", 1)
    for name, res in results.items():
        tp = res["avg_throughput_img_per_sec"]
        speedup = tp / max(baseline_tp, 1)
        mem = res["peak_memory_mb"]
        print(f"{name:<25} {tp:>10.1f}/s {speedup:>9.1f}x {mem:>8.0f}MB")

    # Save combined results
    with open(output_path / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_path}/")


def main():
    parser = argparse.ArgumentParser(description="Training Optimization Benchmark")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output", default="benchmarks/")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = parser.parse_args()

    run_benchmark(args.model, args.dataset, args.batch_size, args.epochs, args.output, args.configs)


if __name__ == "__main__":
    main()
