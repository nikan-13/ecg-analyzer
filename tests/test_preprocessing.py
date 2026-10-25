"""Тесты предобработки: форма, нормализация, воспроизводимость."""
import numpy as np

from src.preprocessing import load_config, normalize_zscore, preprocess

CONFIG = load_config()
NUM_LEADS = CONFIG["signal"]["num_leads"]
NUM_SAMPLES = CONFIG["signal"]["sampling_rate"] * CONFIG["signal"]["duration_sec"]


def make_fake_signal(seed: int = 0) -> np.ndarray:
    """Синтетический сигнал правильной формы для тестов."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(NUM_LEADS, NUM_SAMPLES)).astype(np.float32)


def test_shape_unchanged():
    x = make_fake_signal()
    y = preprocess(x, CONFIG)
    assert y.shape == x.shape


def test_output_dtype_float32():
    y = preprocess(make_fake_signal(), CONFIG)
    assert y.dtype == np.float32


def test_normalization_statistics():
    y = preprocess(make_fake_signal(), CONFIG)
    means = y.mean(axis=-1)
    stds = y.std(axis=-1)
    assert np.allclose(means, 0.0, atol=1e-4)
    assert np.allclose(stds, 1.0, atol=1e-2)


def test_deterministic():
    x = make_fake_signal()
    y1 = preprocess(x, CONFIG)
    y2 = preprocess(x, CONFIG)
    assert np.array_equal(y1, y2)


def test_wrong_number_of_leads_raises():
    x = make_fake_signal()[:5]
    try:
        preprocess(x, CONFIG)
    except ValueError:
        return
    raise AssertionError("Ожидалась ошибка ValueError для неверного числа отведений")


def test_normalize_zscore_unit_std():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=5.0, scale=3.0, size=(3, 500))
    y = normalize_zscore(x)
    assert np.allclose(y.std(axis=-1), 1.0, atol=1e-6)