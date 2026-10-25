"""Подготовка данных PTB-XL: предобработка всех записей и сохранение на диск.

Запуск:
    .venv\Scripts\python.exe -m src.prepare_data
"""
from __future__ import annotations

import numpy as np

from src.dataset import SUPERCLASS_ORDER, load_metadata, load_record
from src.preprocessing import PROJECT_ROOT, load_config, preprocess

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    config = load_config()
    meta = load_metadata()
    n = len(meta)

    num_samples = config["signal"]["sampling_rate"] * config["signal"]["duration_sec"]
    num_leads = config["signal"]["num_leads"]

    X = np.zeros((n, num_leads, num_samples), dtype=np.float32)
    y = meta[SUPERCLASS_ORDER].values.astype(np.float32)
    folds = meta.strat_fold.values.astype(np.int64)

    print(f"загружаем и обрабатываем {n} записей...")
    for i, (_, row) in enumerate(meta.iterrows()):
        raw = load_record(row.filename_lr)     # (12, 1000), сырой сигнал
        X[i] = preprocess(raw, config)          # фильтр + нормализация
        if (i + 1) % 2000 == 0:
            print(f"  обработано {i + 1}/{n}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X.npy", X)
    np.save(PROCESSED_DIR / "y.npy", y)
    np.save(PROCESSED_DIR / "folds.npy", folds)

    print(f"\nготово:")
    print(f"  X     {X.shape}  (~{X.nbytes / 1e9:.1f} ГБ)")
    print(f"  y     {y.shape}  (метки: {SUPERCLASS_ORDER})")
    print(f"  folds {folds.shape}  (уникальные: {sorted(set(folds))})")
    print(f"сохранено в {PROCESSED_DIR}")


if __name__ == "__main__":
    main()