"""Обучение модели на PTB-XL.

Запуск:
    .venv\Scripts\python.exe train.py               # полное обучение
    .venv\Scripts\python.exe train.py --epochs 2    # быстрая проверка, что всё работает
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")  # рисуем графики без окон — сразу в PNG
import matplotlib.pyplot as plt

from src.model import build_model
from src.preprocessing import PROJECT_ROOT, load_config

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def set_seed(seed: int) -> None:
    """Фиксируем случайность — результат воспроизводим."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def compute_val_auc(model, loader, device) -> float:
    """Macro AUC на валидационной выборке."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy()
        preds.append(p)
        targets.append(y.numpy())
    return float(roc_auc_score(np.concatenate(targets), np.concatenate(preds), average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None,
                        help="число эпох (по умолчанию — из config.yaml)")
    args = parser.parse_args()

    config = load_config()
    train_cfg = config["training"]
    class_names = config["model"]["class_names"]
    set_seed(train_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"устройство обучения: {device}")

    # ---------- данные ----------
    print("загрузка данных...")
    X = np.load(PROCESSED_DIR / "X.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    folds = np.load(PROCESSED_DIR / "folds.npy")

    train_mask = np.isin(folds, train_cfg["train_folds"])
    val_mask = np.isin(folds, train_cfg["val_folds"])

    train_ds = TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(y[train_mask]))
    val_ds = TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(y[val_mask]))
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"])
    print(f"  train: {len(train_ds)} записей, val: {len(val_ds)} записей")

    # веса классов: компенсируем, что диагнозы встречаются с разной частотой
    pos = y[train_mask].sum(axis=0)
    neg = train_mask.sum() - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device)
    print("  веса классов:",
          {n: round(float(w), 2) for n, w in zip(class_names, pos_weight)})

    # ---------- модель ----------
    model = build_model(config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    epochs = args.epochs or train_cfg["epochs"]
    history = {"train_loss": [], "val_auc": []}
    best_auc = 0.0
    best_path = MODELS_DIR / "ecg_model.pth"
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"\nобучение: {epochs} эпох\n")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x_batch.size(0)

        train_loss = total_loss / len(train_ds)
        val_auc = compute_val_auc(model, val_loader, device)
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        marker = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            marker = "   * лучшая"

        print(f"эпоха {epoch:2d}/{epochs}   loss {train_loss:.4f}   "
              f"val AUC {val_auc:.4f}   {time.time() - t0:.0f}с{marker}")

    print(f"\nлучшая val AUC: {best_auc:.4f}")
    print(f"веса сохранены: {best_path}")

    # ---------- графики обучения ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["train_loss"], color="steelblue")
    ax1.set_title("Функция потерь")
    ax1.set_xlabel("эпоха")
    ax1.grid(True, alpha=0.3)
    ax2.plot(history["val_auc"], color="darkorange")
    ax2.set_title("Validation AUC")
    ax2.set_xlabel("эпоха")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=120)
    print(f"графики обучения: {RESULTS_DIR / 'training_curves.png'}")

    # ---------- паспорт модели ----------
    card = {
        "architecture": "EcgResNet1D",
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "classes": class_names,
        "val_auc": round(best_auc, 4),
        "epochs_trained": epochs,
        "batch_size": train_cfg["batch_size"],
        "learning_rate": train_cfg["learning_rate"],
        "seed": train_cfg["seed"],
    }
    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=2)
    print(f"паспорт модели: {MODELS_DIR / 'model_card.json'}")


if __name__ == "__main__":
    main()