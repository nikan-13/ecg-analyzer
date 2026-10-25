"""Оценка обученной модели на тестовой выборке (fold 10).

Запуск:
    .venv\Scripts\python.exe evaluate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.model import build_model
from src.preprocessing import PROJECT_ROOT, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


@torch.no_grad()
def main() -> None:
    config = load_config()
    class_names = config["model"]["class_names"]
    threshold = config["predict"]["threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- данные ----------
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    folds = np.load(PROCESSED_DIR / "folds.npy")
    test_mask = np.isin(folds, config["training"]["test_folds"])
    print(f"тестовая выборка: {test_mask.sum()} записей (fold {config['training']['test_folds']})")

    # ---------- модель ----------
    model = build_model(config).to(device)
    model.load_state_dict(
        torch.load(MODELS_DIR / "ecg_model.pth", map_location=device, weights_only=True)
    )
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X[test_mask]), torch.from_numpy(y[test_mask])),
        batch_size=128,
    )
    preds = [torch.sigmoid(model(x.to(device))).cpu().numpy() for x, _ in loader]
    y_pred = np.concatenate(preds)
    y_true = y[test_mask]

    # ---------- метрики ----------
    auc_per_class = roc_auc_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred > threshold, average=None, zero_division=0)

    print(f"\n{'класс':<6} {'AUC':>8} {'F1':>8}")
    print("-" * 24)
    for name, auc, f1 in zip(class_names, auc_per_class, f1_per_class):
        print(f"{name:<6} {auc:>8.4f} {f1:>8.4f}")
    print("-" * 24)
    print(f"{'macro':<6} {auc_per_class.mean():>8.4f} {f1_per_class.mean():>8.4f}")

    metrics = {
        "test_fold": config["training"]["test_folds"],
        "num_test_records": int(test_mask.sum()),
        "threshold": threshold,
        "auc_per_class": {n: round(float(a), 4) for n, a in zip(class_names, auc_per_class)},
        "f1_per_class": {n: round(float(f), 4) for n, f in zip(class_names, f1_per_class)},
        "macro_auc": round(float(auc_per_class.mean()), 4),
        "macro_f1": round(float(f1_per_class.mean()), 4),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nсохранено: {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()