"""Загрузка PTB-XL: сигналы, метки суперклассов, разбиение по folds.

Контракт данных:
    load_metadata() -> DataFrame с колонками filename_lr, strat_fold
                       и пятью метками (NORM, MI, STTC, CD, HYP) 0/1
    load_record(filename) -> np.ndarray формы (12, num_samples), float32
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "ptbxl"

# Порядок суперклассов фиксирован — используется везде: в модели, в UI, в метриках
SUPERCLASS_ORDER = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_metadata(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """Читает метаданные PTB-XL и добавляет метки суперклассов.

    Returns:
        DataFrame с индексом ecg_id и колонками:
        filename_lr, filename_hr, strat_fold, NORM, MI, STTC, CD, HYP
    """
    db = pd.read_csv(data_root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(data_root / "scp_statements.csv", index_col=0)

    # словарь: SCP-код -> суперкласс (только диагностические утверждения)
    diag = scp[scp.diagnostic == 1]
    superclass_map = diag.diagnostic_class.dropna().to_dict()

    def to_superclass_labels(scp_codes: str) -> pd.Series:
        codes = ast.literal_eval(scp_codes)  # строка вида "{'NORM': 100.0}"
        classes = {superclass_map[c] for c in codes if c in superclass_map}
        return pd.Series(
            {name: int(name in classes) for name in SUPERCLASS_ORDER}
        )

    labels = db.scp_codes.apply(to_superclass_labels)
    return pd.concat(
        [db[["filename_lr", "filename_hr", "strat_fold"]], labels],
        axis=1,
    )


def load_record(
    filename: str,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> np.ndarray:
    """Загружает одну запись ЭКГ из WFDB-формата.

    Args:
        filename: путь внутри датасета, например "records100/00000/00001_lr".

    Returns:
        Сигнал формы (12, num_samples), float32.
    """
    import wfdb

    signal, _ = wfdb.rdsamp(str(data_root / filename))
    return signal.T.astype(np.float32)