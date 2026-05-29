"""데이터 파이프라인 검증."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import (
    DATA_PROCESSED_DIR,
    LEAKAGE_COLS,
    TARGET_COL,
)


def test_processed_files_exist() -> None:
    for name in ("train", "val", "test"):
        assert (DATA_PROCESSED_DIR / f"{name}.parquet").exists()


def test_no_leakage_columns_in_processed() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "train.parquet")
    leaked = set(df.columns) & set(LEAKAGE_COLS)
    assert not leaked, f"Leakage columns leaked into processed: {leaked}"


def test_target_is_binary() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "train.parquet")
    assert TARGET_COL in df.columns
    assert set(df[TARGET_COL].unique()) <= {0, 1}


def test_positive_rate_in_expected_range() -> None:
    df = pd.read_parquet(DATA_PROCESSED_DIR / "test.parquet")
    p = df[TARGET_COL].mean()
    # Lending Club default rate is around 15-25% after our label mapping
    assert 0.10 < p < 0.35, f"Unexpected positive rate: {p}"
