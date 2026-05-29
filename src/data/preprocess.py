"""Raw Lending Club CSV → cleaned, leakage-free train/val/test parquet.

사용:
    python -m src.data.preprocess --sample 200000  # 빠른 실험
    python -m src.data.preprocess                  # 전체 데이터
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    DATA_SOURCE_DIR,
    LABEL_DEFAULT,
    LABEL_DROP,
    LEAKAGE_COLS,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    RAW_CSV_NAME,
    TARGET_COL,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("preprocess")


def _emp_length_to_int(val: object) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"< 1 year", "<1 year"}:
        return 0.0
    if s in {"10+ years", "10+ year"}:
        return 10.0
    digits = "".join(c for c in s if c.isdigit())
    return float(digits) if digits else np.nan


def _term_to_int(val: object) -> str:
    if pd.isna(val):
        return "unknown"
    return str(val).strip()


def _credit_history_years(earliest_cr_line: pd.Series, issue_d: pd.Series) -> pd.Series:
    ecl = pd.to_datetime(earliest_cr_line, format="%b-%Y", errors="coerce")
    iss = pd.to_datetime(issue_d, format="%b-%Y", errors="coerce")
    diff = (iss - ecl).dt.days / 365.25
    return diff.clip(lower=0)


def load_raw(sample: int | None = None) -> pd.DataFrame:
    src = DATA_SOURCE_DIR / RAW_CSV_NAME
    log.info("Reading raw CSV from %s", src)
    # We only need a defined subset of columns.
    needed = set(ALL_FEATURES) | {"loan_status", "earliest_cr_line", "issue_d"}
    # We don't yet know exact column names — read header first.
    header = pd.read_csv(src, nrows=0).columns.tolist()
    usecols = [c for c in header if c in needed]
    log.info("Reading %d columns (of %d)", len(usecols), len(header))
    df = pd.read_csv(src, usecols=usecols, low_memory=False)
    log.info("Raw shape: %s", df.shape)
    if sample is not None and sample < len(df):
        df = df.sample(n=sample, random_state=RANDOM_SEED).reset_index(drop=True)
        log.info("Sampled to: %s", df.shape)
    return df


def make_target(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[~df["loan_status"].isin(LABEL_DROP)].copy()
    df[TARGET_COL] = df["loan_status"].map(LABEL_DEFAULT)
    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    log.info(
        "Label mapping: kept %d / %d rows (%.1f%% positive)",
        len(df), before, 100 * df[TARGET_COL].mean(),
    )
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].apply(_emp_length_to_int).astype(float)
        # cast back to string buckets for categorical handling
        df["emp_length"] = df["emp_length"].fillna(-1).astype(int).astype(str)
    if "term" in df.columns:
        df["term"] = df["term"].apply(_term_to_int)
    if "earliest_cr_line" in df.columns and "issue_d" in df.columns:
        df["credit_history_years"] = _credit_history_years(
            df["earliest_cr_line"], df["issue_d"]
        )
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)
    # Reasonable cap on revol_util / dti
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].clip(lower=0, upper=200)
    if "dti" in df.columns:
        df["dti"] = df["dti"].clip(lower=-10, upper=200)
    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ALL_FEATURES if c in df.columns] + [TARGET_COL]
    missing = set(ALL_FEATURES) - set(df.columns)
    if missing:
        log.warning("Missing expected columns (will be skipped): %s", sorted(missing))
    return df[keep].copy()


def split_and_save(df: pd.DataFrame) -> dict[str, Path]:
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df[TARGET_COL], random_state=RANDOM_SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp[TARGET_COL], random_state=RANDOM_SEED
    )
    ensure_dirs()
    paths = {}
    for name, part in {"train": train, "val": val, "test": test}.items():
        p = DATA_PROCESSED_DIR / f"{name}.parquet"
        part.to_parquet(p, index=False)
        paths[name] = p
        log.info(
            "Saved %s: %d rows, %.2f%% positive → %s",
            name, len(part), 100 * part[TARGET_COL].mean(), p,
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="If set, use a random subsample of this many rows (for fast iteration).",
    )
    args = parser.parse_args()

    df = load_raw(sample=args.sample)
    df = make_target(df)
    df = engineer_features(df)
    # Drop leakage columns AFTER feature engineering (issue_d is needed for credit_history_years)
    df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns], errors="ignore")
    df = select_columns(df)
    df = df.dropna(subset=["loan_amnt", "annual_inc"]).reset_index(drop=True)
    split_and_save(df)
    log.info("Done.")


if __name__ == "__main__":
    main()
