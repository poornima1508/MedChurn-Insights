
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# DATASET 6: SYNTHETIC MEDICAID ADMINISTRATIVE HISTORY
# ============================================================

SEED = 20260930
rng = np.random.default_rng(SEED + 5)

RAW_DIR = Path("data/raw")

# ============================================================
# 1. LOAD EXISTING DATASETS
# ============================================================

beneficiaries = pd.read_csv(
    RAW_DIR / "02_beneficiaries.csv"
)

households = pd.read_csv(
    RAW_DIR / "01_households.csv"
)

renewals = pd.read_csv(
    RAW_DIR / "04_renewal_episodes.csv"
)

# Handle household column naming issue
if "index" in households.columns:
    households = households.rename(
        columns={"index": "household_id"}
    )

# ============================================================
# 2. MERGE EXISTING INFORMATION
# ============================================================

data = renewals[
    ["episode_id", "beneficiary_id", "prior_coverage_gaps"]
].merge(
    beneficiaries[
        ["beneficiary_id", "household_id", "eligibility_category"]
    ],
    on="beneficiary_id",
    how="left",
    validate="one_to_one"
).merge(
    households[
        ["household_id", "income_change_pct"]
    ],
    on="household_id",
    how="left",
    validate="many_to_one"
)

n = len(data)

assert n == 30000, "Expected 30,000 records"

assert data[
    ["household_id", "income_change_pct"]
].notna().all().all(), "Missing household information"

# ============================================================
# 3. GENERATE PRIOR DOCUMENTATION ISSUES
# ============================================================

# Historical coverage gaps increase the simulated probability
# of having experienced documentation problems.

documentation_probability = np.clip(
    0.10 + 0.09 * data["prior_coverage_gaps"].to_numpy(),
    0,
    0.70
)

prior_documentation_issues = rng.binomial(
    1,
    documentation_probability
)

# ============================================================
# 4. GENERATE INCOME VERIFICATION REQUIREMENT
# ============================================================

# Larger income changes increase the simulated probability
# of requiring additional verification.

# Convert stored percentage points into a decimal proportion.
# Example: 20.0 (%) becomes 0.20.
income_change = np.abs(
    data["income_change_pct"].to_numpy() / 100.0
)

verification_probability = np.clip(
    0.15 + 0.50 * income_change,
    0.05,
    0.80
)

income_verification_required = rng.binomial(
    1,
    verification_probability
)

# ============================================================
# 5. CALCULATE ADMINISTRATIVE COMPLEXITY
# ============================================================

complexity_score = (
    prior_documentation_issues
    + income_verification_required
    + (data["prior_coverage_gaps"].to_numpy() > 0).astype(int)
)

administrative_complexity = np.select(
    [
        complexity_score == 0,
        complexity_score == 1,
        complexity_score >= 2
    ],
    ["Low", "Medium", "High"],
    default="Unknown"
)

# ============================================================
# 6. CREATE DATASET
# ============================================================

administrative = pd.DataFrame({
    "episode_id": data["episode_id"],
    "beneficiary_id": data["beneficiary_id"],
    "prior_documentation_issues": prior_documentation_issues,
    "income_verification_required": income_verification_required,
    "administrative_complexity": administrative_complexity
})

# ============================================================
# 7. VALIDATE DATASET
# ============================================================

assert len(administrative) == 30000
assert administrative["episode_id"].is_unique
assert administrative["beneficiary_id"].is_unique
assert administrative.isnull().sum().sum() == 0

assert administrative[
    "administrative_complexity"
].isin(["Low", "Medium", "High"]).all()

assert administrative[
    "prior_documentation_issues"
].isin([0, 1]).all()

assert administrative[
    "income_verification_required"
].isin([0, 1]).all()

print("\nAll administrative history validation checks passed!")

# ============================================================
# 8. SAVE DATASET
# ============================================================

output_file = RAW_DIR / "06_administrative_history.csv"

administrative.to_csv(
    output_file,
    index=False
)

print("\nAdministrative history dataset generated successfully!")
print("Saved to:", output_file)

print("\nNumber of records:", len(administrative))
print("Number of columns:", len(administrative.columns))

print("\nFirst five rows:")
print(administrative.head().to_string(index=False))

print("\nAdministrative complexity distribution:")
print(administrative["administrative_complexity"].value_counts())

print("\nPrior documentation issues:")
print(administrative["prior_documentation_issues"].value_counts())

print("\nIncome verification required:")
print(administrative["income_verification_required"].value_counts())

print("\nMissing values:")
print(administrative.isnull().sum())

print("\nAdministrative history validation: PASSED")
