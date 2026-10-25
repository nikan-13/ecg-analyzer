"""Инференс: файл ЭКГ → вероятности диагнозов.

Использует ТОТ ЖЕ preprocessing, что и при обучении —
это гарантирует, что модель получит данные в привычном виде.

Контракт:
    load_signal_from_csv(path) -> np.ndarray формы (12, 1000), float32
    predict_signal(signal, model, config) -> dict с вероятностями
    predict_file(path, model, config) -> dict (удобная обёртка)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import build_model
from src.preprocessing import load_config, preprocess

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "ecg_model.pth"


def load_signal_from_csv(path: str | Path) -> np.ndarray:
    """Читает CSV с 12 колонками (отведения) и возвращает сигнал (12, num_samples).

    Ожидаемый формат CSV:
        I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
        -0.021,0.043,...
        ...
    """
    df = pd.read_csv(path)

    if df.shape[1] != 12:
        raise ValueError(
            f"Ожидается 12 столбцов (отведения), получено {df.shape[1]}. "
            f"Столбцы файла: {list(df.columns)}"
        )

    signal = df.values.T.astype(np.float32)  # (12, num_samples)
    return signal


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    config: dict | None = None,
) -> torch.nn.Module:
    """Загружает обученную модель из файла весов."""
    if config is None:
        config = load_config()

    model = build_model(config)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_signal(
    signal: np.ndarray,
    model: torch.nn.Module,
    config: dict | None = None,
) -> dict[str, float]:
    """Сигнал ЭКГ → словарь {класс: вероятность}.

    Args:
        signal: массив формы (12, num_samples), сырой или уже обработанный.
        model: загруженная модель (см. load_model).
        config: настройки; если None — из config.yaml.
    """
    if config is None:
        config = load_config()

    # та же предобработка, что при обучении!
    x = preprocess(signal, config)

    # добавляем batch-ось: (12, 1000) -> (1, 12, 1000)
    x_tensor = torch.from_numpy(x).unsqueeze(0)

    logits = model(x_tensor)
    probs = torch.sigmoid(logits).squeeze(0).numpy()

    class_names = config["model"]["class_names"]
    return {name: round(float(p), 4) for name, p in zip(class_names, probs)}


def predict_file(
    path: str | Path,
    model: torch.nn.Module,
    config: dict | None = None,
) -> dict[str, float]:
    """Файл CSV → вероятности диагнозов. Удобная обёртка."""
    signal = load_signal_from_csv(path)
    return predict_signal(signal, model, config)