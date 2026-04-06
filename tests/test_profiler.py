"""Tests for training profiler components."""
import pytest
import torch
import numpy as np

from src.optimizations import OptimizationConfig, apply_optimizations
from src.models import get_model, list_models
from src.trainer import Trainer, TrainingMetrics


class TestOptimizationConfig:
    def test_baseline(self):
        cfg = OptimizationConfig.baseline()
        assert cfg.amp is False
        assert cfg.compile is False

    def test_amp_only(self):
        cfg = OptimizationConfig.amp_only()
        assert cfg.amp is True
        assert cfg.compile is False

    def test_all_combined(self):
        cfg = OptimizationConfig.all_combined()
        assert cfg.amp is True
        assert cfg.compile is True
        assert cfg.pin_memory is True
        assert cfg.num_workers == 4


class TestModels:
    def test_list_models(self):
        models = list_models()
        assert "resnet50" in models
        assert "resnet18" in models

    def test_get_resnet18(self):
        model = get_model("resnet18")
        assert isinstance(model, torch.nn.Module)
        # Verify forward pass works
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 1000)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_model("nonexistent_model")


class TestTrainingMetrics:
    def test_empty_metrics(self):
        m = TrainingMetrics()
        assert m.avg_throughput == 0
        assert m.avg_epoch_time == 0

    def test_summary(self):
        m = TrainingMetrics()
        m.epoch_times = [10.0, 12.0]
        m.losses = [0.5, 0.3]
        m.throughput_samples_per_sec = [500.0, 600.0]
        m.peak_memory_mb = 2048.0
        m.total_time_sec = 22.0

        s = m.summary()
        assert s["avg_throughput_img_per_sec"] == 550.0
        assert s["total_time_sec"] == 22.0
        assert s["final_loss"] == 0.3


class TestApplyOptimizations:
    def test_baseline_no_changes(self):
        model = get_model("resnet18")
        cfg = OptimizationConfig.baseline()
        model, scaler, compiled = apply_optimizations(model, cfg, torch.device("cpu"))
        assert scaler is None
        assert compiled is False

    def test_amp_creates_scaler(self):
        model = get_model("resnet18")
        cfg = OptimizationConfig.amp_only()
        model, scaler, compiled = apply_optimizations(model, cfg, torch.device("cpu"))
        assert scaler is not None
