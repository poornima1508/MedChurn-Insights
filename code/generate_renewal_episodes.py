
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Synthetic Medicaid Renewal Episodes Generator
# ============================================================

SEED = 20260930
rng = np.random.default_rng(SEED + 3)

RAW_DIR = Path("data/raw")

beneficiaries = pd.read_csv(
    RAW_DIR / "02_beneficiaries.csv"
)

n = len(beneficiaries)

# ============================================================
# 1. GENERATE RENEWAL DUE DATES
# ============================================================

start_date = pd.Timestamp("2025-01-01")

random_days = rng.integers(
    0, 365, size=n
)

renewal_due_dates = (
    start_date + pd.to_timedelta(random_days, unit="D")
)

# Prediction occurs exactly 30 days before renewal
prediction_dates = renewal_due_dates - pd.Timedelta(days=30)

# ============================================================
# 2. GENERATE COVERAGE HISTORY
# ============================================================

# Coverage begins 6 to 36 months before prediction.
# This is a synthetic assumption, not observed agency data.

months_covered = rng.integers(
    6, 37, size=n
)

coverage_start_dates = (
    prediction_dates -
    pd.to_timedelta(months_covered * 30, unit="D")
)

# Number of prior coverage gaps known at prediction time
prior_coverage_gaps = rng.choice(
    [0, 1, 2, 3],
    size=n,
    p=[0.73, 0.18, 0.07, 0.02]
)

# Number of previous renewals
previous_renewals = np.maximum(
    (months_covered // 12) - 1,
    0
)

# ============================================================
# 3. CREATE RENEWAL DATASET
# ============================================================

renewals = pd.DataFrame({
    "episode_id": [
        f"R{i:07d}" for i in range(1, n + 1)
    ],
    "beneficiary_id": beneficiaries["beneficiary_id"],
    "renewal_due_date": renewal_due_dates,
    "prediction_date": prediction_dates,
    "coverage_start_date": coverage_start_dates,
    "months_covered": months_covered,
    "prior_coverage_gaps": prior_coverage_gaps,
    "previous_renewals": previous_renewals
})

# ============================================================
# 4. VALIDATION
# ============================================================

assert len(renewals) == 30000
assert renewals["episode_id"].is_unique
assert renewals["beneficiary_id"].is_unique

assert (
    renewals["renewal_due_date"] -
    renewals["prediction_date"]
).dt.days.eq(30).all()

assert (
    renewals["coverage_start_date"] <=
    renewals["prediction_date"]
).all()

assert renewals.isnull().sum().sum() == 0

# ============================================================
# 5. SAVE
# ============================================================

output_file = RAW_DIR / "04_renewal_episodes.csv"

renewals.to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)

print("Renewal episodes generated successfully!")
print("Number of episodes:", len(renewals))
print("Number of columns:", len(renewals.columns))

print("\nFirst five rows:")
print(renewals.head().to_string(index=False))

print("\nRenewal date range:")
print(
    renewals["renewal_due_date"].min(),
    "to",
    renewals["renewal_due_date"].max()
)

print("\nPrior coverage gaps:")
print(renewals["prior_coverage_gaps"].value_counts())

print("\nMissing values:")
print(renewals.isnull().sum())

print("\nRenewal episode validation: PASSED")
