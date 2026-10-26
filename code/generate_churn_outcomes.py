
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# DATASET 7: SYNTHETIC MEDICAID CHURN OUTCOMES
# ============================================================

SEED = 20260930
rng = np.random.default_rng(SEED + 6)

RAW_DIR = Path("data/raw")

# ============================================================
# 1. LOAD DATASETS
# ============================================================

renewals = pd.read_csv(
    RAW_DIR / "04_renewal_episodes.csv",
    parse_dates=["renewal_due_date", "prediction_date"]
)

outreach = pd.read_csv(
    RAW_DIR / "05_renewal_outreach.csv"
)

administrative = pd.read_csv(
    RAW_DIR / "06_administrative_history.csv"
)

# ============================================================
# 2. MERGE PREDICTION FEATURES
# ============================================================

data = renewals.merge(
    outreach.drop(columns=["beneficiary_id"]),
    on="episode_id",
    validate="one_to_one"
).merge(
    administrative.drop(columns=["beneficiary_id"]),
    on="episode_id",
    validate="one_to_one"
)

assert len(data) == 30000
assert data.isnull().sum().sum() == 0

# ============================================================
# 3. SIMULATE CHURN PROBABILITY
# ============================================================

# All coefficients below are hypothetical simulation
# assumptions, not estimated from actual Medicaid records.

complexity_risk = data[
    "administrative_complexity"
].map({
    "Low": 0.00,
    "Medium": 0.35,
    "High": 0.75
}).to_numpy()

risk_score = (
    -2.20
    + complexity_risk
    + 0.30 * data["prior_coverage_gaps"].to_numpy()
    + 0.55 * data["notice_returned"].to_numpy()
    + 0.40 * (1 - data["notice_sent"].to_numpy())
    + 0.25 * data["prior_documentation_issues"].to_numpy()
    - 0.50 * data["responded_pre_prediction"].to_numpy()
)

# Logistic transformation
churn_probability = 1 / (1 + np.exp(-risk_score))

# Draw simulated binary outcomes
churn_30d = rng.binomial(
    1,
    churn_probability
)

# ============================================================
# 4. CREATE OUTCOME DATASET
# ============================================================

outcomes = pd.DataFrame({
    "episode_id": data["episode_id"],
    "beneficiary_id": data["beneficiary_id"],
    "renewal_due_date": data["renewal_due_date"],
    "outcome_window_end": (
        data["renewal_due_date"] + pd.Timedelta(days=30)
    ),
    "churn_30d": churn_30d
})

# ============================================================
# 5. VALIDATE
# ============================================================

assert len(outcomes) == 30000
assert outcomes["episode_id"].is_unique
assert outcomes["beneficiary_id"].is_unique
assert outcomes.isnull().sum().sum() == 0
assert outcomes["churn_30d"].isin([0, 1]).all()

assert (
    outcomes["outcome_window_end"] -
    outcomes["renewal_due_date"]
).dt.days.eq(30).all()

# ============================================================
# 6. SAVE
# ============================================================

output_file = RAW_DIR / "07_churn_outcomes.csv"

outcomes.to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)

print("\nChurn outcome dataset generated successfully!")
print("Saved to:", output_file)

print("\nNumber of records:", len(outcomes))
print("Number of columns:", len(outcomes.columns))

print("\nChurn distribution:")
print(outcomes["churn_30d"].value_counts())

print("\nOverall churn rate:")
print(f"{outcomes['churn_30d'].mean() * 100:.2f}%")

print("\nFirst five rows:")
print(outcomes.head().to_string(index=False))

print("\nMissing values:")
print(outcomes.isnull().sum())

print("\nChurn outcome validation: PASSED")
