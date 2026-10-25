"""Экспорт демо-записей ЭКГ из PTB-XL в маленькие CSV.

Зачем: приложению и тестам не нужен весь датасет в 2 ГБ — достаточно
нескольких записей по 12 отведений. Результат лежит в data/samples/
и попадает в git (файлы маленькие).

Запуск:
    .venv\Scripts\python.exe -m src.export_samples
"""
from __future__ import annotations

import json

import pandas as pd

from src.dataset import (
    PROJECT_ROOT,
    SUPERCLASS_ORDER,
    load_metadata,
    load_record,
)
from src.preprocessing import load_config

SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
CONFIG = load_config()
LEAD_NAMES = CONFIG["signal"]["lead_names"]


def main(records_per_class: int = 1) -> None:
    """Выбирает по несколько записей на каждый суперкласс и сохраняет в CSV."""
    meta = load_metadata()
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    picked = (
        pd.concat(
            [meta[meta[cls] == 1].head(records_per_class) for cls in SUPERCLASS_ORDER]
        )
        .drop_duplicates(subset="filename_lr")
    )

    index = {}
    for ecg_id, row in picked.iterrows():
        signal = load_record(row.filename_lr)          # (12, 1000)
        df = pd.DataFrame(signal.T, columns=LEAD_NAMES)  # строки = отсчёты
        out_path = SAMPLES_DIR / f"demo_{ecg_id:05d}.csv"
        df.to_csv(out_path, index=False)

        labels = {
            cls: int(row[cls]) for cls in SUPERCLASS_ORDER if row[cls] == 1
        }
        index[out_path.name] = {
            "ecg_id": int(ecg_id),
            "labels": labels,
            "fold": int(row.strat_fold),
        }
        print(f"сохранено: {out_path.name}  метки: {list(labels)}")

    with open(SAMPLES_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print(f"\nготово: {len(index)} записей в {SAMPLES_DIR}")


if __name__ == "__main__":
    main()