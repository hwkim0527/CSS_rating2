"""Train baseline (logistic) + XGBoost on processed parquet.

Outputs:
  artifacts/baseline_logistic.joblib  (preprocessor + LR)
  artifacts/xgboost.joblib            (preprocessor + booster)
  artifacts/metrics.json              (test-set metrics, used by /api/compare)
  artifacts/feature_schema.json       (numeric/categorical lists)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    METRICS_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COL,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("train")


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic — separation between distributions."""
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    pos_cum = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    neg_cum = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
    return float(np.max(np.abs(pos_cum - neg_cum)))


def load_splits() -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "val", "test"):
        p = DATA_PROCESSED_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run preprocess first.")
        out[name] = pd.read_parquet(p)
    return out


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    X = df[available_num + available_cat].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=50,
                    max_categories=30,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric),
            ("cat", cat_pipe, categorical),
        ],
        remainder="drop",
    )


def evaluate(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ks": ks_statistic(y_true, y_prob),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "positive_rate": float(y_true.mean()),
    }
    log.info(
        "%s — AUC=%.4f KS=%.4f AP=%.4f Acc=%.4f F1=%.4f",
        name,
        metrics["auc"],
        metrics["ks"],
        metrics["average_precision"],
        metrics["accuracy"],
        metrics["f1"],
    )
    return metrics


def train_logistic(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    pre = build_preprocessor(numeric, categorical)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    numeric: list[str],
    categorical: list[str],
) -> Pipeline:
    pre = build_preprocessor(numeric, categorical)
    # Fit transformer once, then train booster directly to use eval set.
    X_train_t = pre.fit_transform(X_train, y_train)
    X_val_t = pre.transform(X_val)
    scale_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    booster = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="auc",
        early_stopping_rounds=30,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    booster.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    log.info("XGB best iter=%s best AUC=%.4f", booster.best_iteration, booster.best_score)
    pipe = Pipeline(steps=[("pre", pre), ("clf", booster)])
    return pipe


def main() -> None:
    ensure_dirs()
    splits = load_splits()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    numeric = [c for c in NUMERIC_FEATURES if c in X_train.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    log.info("Numeric (%d): %s", len(numeric), numeric)
    log.info("Categorical (%d): %s", len(categorical), categorical)

    log.info("=== Training Logistic Regression (baseline) ===")
    lr = train_logistic(X_train, y_train, numeric, categorical)
    lr_test_prob = lr.predict_proba(X_test)[:, 1]
    lr_metrics = evaluate("Logistic[test]", y_test, lr_test_prob)

    log.info("=== Training XGBoost ===")
    xgb = train_xgb(X_train, y_train, X_val, y_val, numeric, categorical)
    xgb_test_prob = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate("XGBoost[test]", y_test, xgb_test_prob)

    # Persist artifacts
    joblib.dump(lr, ARTIFACTS_DIR / "baseline_logistic.joblib")
    joblib.dump(xgb, ARTIFACTS_DIR / "xgboost.joblib")
    log.info("Saved models to %s", ARTIFACTS_DIR)

    schema = {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "target": TARGET_COL,
    }
    (ARTIFACTS_DIR / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Metrics payload (consumed by /api/compare)
    auc_delta = xgb_metrics["auc"] - lr_metrics["auc"]
    payload = {
        "dataset": {
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "positive_rate": float(y_test.mean()),
        },
        "models": {
            "logistic_regression": {
                "label_kr": "로지스틱 회귀 (전통적 모델)",
                **lr_metrics,
            },
            "xgboost": {
                "label_kr": "XGBoost (그래디언트 부스팅)",
                **xgb_metrics,
            },
            "llm_qwen25_7b": {
                "label_kr": "Qwen2.5-7B QLoRA (LLM, GCP 학습 대기)",
                "status": "pending_training",
                "auc": None,
                "ks": None,
                "average_precision": None,
                "accuracy": None,
                "f1": None,
                "positive_rate": None,
            },
        },
        "comparison_vs_baseline": {
            "xgboost_auc_delta": auc_delta,
            "xgboost_auc_pct_improvement": (auc_delta / lr_metrics["auc"]) * 100,
            "target_min_auc_delta": 0.05,
            "meets_10pct_goal": auc_delta >= 0.05,
        },
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Wrote %s", METRICS_PATH)


if __name__ == "__main__":
    main()
