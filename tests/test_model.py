"""Тесты модели: форма выхода, диапазон значений, обучаемость."""
import torch

from src.model import build_model
from src.preprocessing import load_config

CONFIG = load_config()
NUM_LEADS = CONFIG["signal"]["num_leads"]
NUM_SAMPLES = CONFIG["signal"]["sampling_rate"] * CONFIG["signal"]["duration_sec"]
NUM_CLASSES = CONFIG["model"]["num_classes"]


def test_output_shape():
    model = build_model(CONFIG)
    x = torch.randn(2, NUM_LEADS, NUM_SAMPLES)  # batch из 2 записей
    y = model(x)
    assert y.shape == (2, NUM_CLASSES)


def test_output_is_finite():
    model = build_model(CONFIG)
    x = torch.randn(1, NUM_LEADS, NUM_SAMPLES)
    y = model(x)
    assert torch.isfinite(y).all()


def test_single_record():
    model = build_model(CONFIG)
    x = torch.randn(1, NUM_LEADS, NUM_SAMPLES)
    y = model(x)
    assert y.shape == (1, NUM_CLASSES)


def test_model_is_trainable():
    """Градиенты должны течь — иначе модель не обучится."""
    model = build_model(CONFIG)
    x = torch.randn(2, NUM_LEADS, NUM_SAMPLES)
    target = torch.zeros(2, NUM_CLASSES)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0