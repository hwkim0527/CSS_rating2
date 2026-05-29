"""중앙 설정. 경로와 컬럼 정의를 한 곳에서 관리한다."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT
DATA_SOURCE_DIR = (PROJECT_ROOT.parent / "data_source").resolve()
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RAW_CSV_NAME = "loan.csv"
DICT_XLSX_NAME = "LCDataDictionary.xlsx"

RANDOM_SEED = 42

# === Label mapping ===
LABEL_DEFAULT = {
    "Charged Off": 1,
    "Default": 1,
    "Late (31-120 days)": 1,
    "Late (16-30 days)": 1,
    "Fully Paid": 0,
}
LABEL_DROP = {
    "Current",
    "In Grace Period",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}

# === Leakage columns — drop unconditionally ===
LEAKAGE_COLS = [
    # Post-funding payment info
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "out_prncp", "out_prncp_inv",
    # Lending Club's own grades (avoid label leakage / tautology)
    "grade", "sub_grade",
    # Post-issuance credit pulls
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    # ID / URL / free text
    "id", "member_id", "url", "desc", "title", "emp_title", "zip_code",
    # Hardship/settlement (only present for distressed loans)
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    # Joint apps that don't apply to single applicants (drop to keep model simple)
    "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
    "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
    "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
    # Date columns we transform separately
    "issue_d",
]

# === Feature spec ===
NUMERIC_FEATURES = [
    "loan_amnt", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies",
    "credit_history_years",  # engineered
    "int_rate",  # keep as a *feature* (it reflects market pricing of risk at app)
]
CATEGORICAL_FEATURES = [
    "term", "home_ownership", "verification_status",
    "purpose", "addr_state", "emp_length", "application_type",
    "initial_list_status",
]
TARGET_COL = "default"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_dirs() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
