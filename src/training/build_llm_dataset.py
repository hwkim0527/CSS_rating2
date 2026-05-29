"""processed parquet → SFT JSONL for LLM fine-tuning.

사용:
    python -m src.training.build_llm_dataset --train-samples 50000 --val-samples 5000
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.training.serialize import dataframe_to_jsonl
from src.utils.config import DATA_PROCESSED_DIR, REPO_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_llm_dataset")

OUT_DIR = REPO_ROOT / "data" / "llm"


def stratified_sample(df: pd.DataFrame, n: int, target: str = "default") -> pd.DataFrame:
    if len(df) <= n:
        return df
    pos_rate = df[target].mean()
    n_pos = int(round(n * pos_rate))
    n_neg = n - n_pos
    pos = df[df[target] == 1].sample(min(n_pos, (df[target] == 1).sum()), random_state=42)
    neg = df[df[target] == 0].sample(min(n_neg, (df[target] == 0).sum()), random_state=42)
    return pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=50000)
    parser.add_argument("--val-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, n in [
        ("train", args.train_samples),
        ("val", args.val_samples),
        ("test", args.test_samples),
    ]:
        df = pd.read_parquet(DATA_PROCESSED_DIR / f"{split}.parquet")
        sampled = stratified_sample(df, n)
        records = dataframe_to_jsonl(sampled)
        write_jsonl(records, OUT_DIR / f"{split}.jsonl")


if __name__ == "__main__":
    main()
