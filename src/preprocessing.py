"""Предобработка сигнала ЭКГ.

Единая точка правды: этот код используется и при обучении модели,
и при анализе новых записей. Любое изменение здесь меняет поведение
и обучения, и предсказания одновременно.

Контракт данных:
    вход  — np.ndarray формы (num_leads, num_samples), float
    выход — та же форма, float32, отфильтрованный и нормализованный сигнал
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.signal import butter, filtfilt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Загружает настройки проекта из config.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 2,
) -> np.ndarray:
    """Полосовой фильтр Баттерворта: убирает дрейф базовой линии и шум.

    Args:
        signal: массив формы (num_leads, num_samples).
        fs: частота дискретизации, Гц.
        low, high: границы полосы пропускания, Гц.
        order: порядок фильтра.
    """
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def normalize_zscore(signal: np.ndarray) -> np.ndarray:
    """Стандартизация каждого отведения: нулевое среднее, единичное СКО."""
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / (std + 1e-8)


def preprocess(signal: np.ndarray, config: dict | None = None) -> np.ndarray:
    """Полный конвейер предобработки ЭКГ.

    Args:
        signal: сырой сигнал формы (num_leads, num_samples).
        config: настройки из config.yaml; если None — загружаются автоматически.

    Returns:
        Готовый для модели массив (num_leads, num_samples), dtype float32.
    """
    if config is None:
        config = load_config()

    sig_cfg = config["signal"]
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(
            f"Ожидается массив 2D (отведения, отсчёты), получено ndim={x.ndim}"
        )

    expected_leads = sig_cfg["num_leads"]
    if x.shape[0] != expected_leads:
        raise ValueError(
            f"Ожидается {expected_leads} отведений, получено {x.shape[0]}"
        )

    low, high = sig_cfg["bandpass"]
    x = bandpass_filter(x, fs=sig_cfg["sampling_rate"], low=low, high=high)

    if sig_cfg["normalize"] == "zscore":
        x = normalize_zscore(x)

    return x.astype(np.float32)