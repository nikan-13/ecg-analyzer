"""Тесты инференса: загрузка CSV, форма результата, диапазон вероятностей."""
import numpy as np
import torch

from src.model import build_model
from src.predict import load_signal_from_csv, predict_signal
from src.preprocessing import load_config, PROJECT_ROOT

CONFIG = load_config()
NUM_LEADS = CONFIG["signal"]["num_leads"]
NUM_SAMPLES = CONFIG["signal"]["sampling_rate"] * CONFIG["signal"]["duration_sec"]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def test_load_demo_csv():
    """Первая демо-запись должна загружаться с правильной формой."""
    csv_files = sorted(SAMPLES_DIR.glob("demo_*.csv"))
    assert len(csv_files) > 0, "нет демо-файлов, запусти src.export_samples"

    signal = load_signal_from_csv(csv_files[0])
    assert signal.shape == (NUM_LEADS, NUM_SAMPLES)
    assert signal.dtype == np.float32

def test_predict_returns_all_classes():
    """Вероятности должны быть для каждого класса, в диапазоне [0, 1]."""
    model = build_model(CONFIG)
    model.eval()  # режим инференса: dropout выключен, результат воспроизводим
    x = np.random.default_rng(0).normal(
        size=(NUM_LEADS, NUM_SAMPLES)
    ).astype(np.float32)

    result = predict_signal(x, model, CONFIG)

    assert set(result.keys()) == set(CONFIG["model"]["class_names"])
    for name, prob in result.items():
        assert 0.0 <= prob <= 1.0, f"{name}: {prob} вне [0, 1]"


def test_predict_deterministic():
    """Одинаковый вход → одинаковый результат (в режиме eval)."""
    model = build_model(CONFIG)
    model.eval()  # ключевая строка: без неё dropout даёт разные результаты
    x = np.random.default_rng(42).normal(
        size=(NUM_LEADS, NUM_SAMPLES)
    ).astype(np.float32)

    r1 = predict_signal(x, model, CONFIG)
    r2 = predict_signal(x, model, CONFIG)
    assert r1 == r2

def test_wrong_columns_raises():
    """CSV с неправильным числом столбцов должен вызывать ошибку."""
    import tempfile, pandas as pd

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        pd.DataFrame(np.random.randn(100, 5)).to_csv(f, index=False)
        try:
            load_signal_from_csv(f.name)
        except ValueError:
            return
    raise AssertionError("Ожидалась ValueError для файла с 5 столбцами")