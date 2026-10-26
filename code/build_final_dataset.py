
from pathlib import Path
import pandas as pd

# ============================================================
# STEP 8: BUILD FINAL MEDICAID MODELING DATASET
# ============================================================

RAW_DIR = Path("data/raw")
FINAL_DIR = Path("data/final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load datasets
households = pd.read_csv(RAW_DIR / "01_households.csv")
beneficiaries = pd.read_csv(RAW_DIR / "02_beneficiaries.csv")
fairness = pd.read_csv(RAW_DIR / "03_fairness_attributes.csv")
renewals = pd.read_csv(RAW_DIR / "04_renewal_episodes.csv")
outreach = pd.read_csv(RAW_DIR / "05_renewal_outreach.csv")
administrative = pd.read_csv(
    RAW_DIR / "06_administrative_history.csv"
)
outcomes = pd.read_csv(RAW_DIR / "07_churn_outcomes.csv")

# Fix old household column name if necessary
if "index" in households.columns:
    households = households.rename(
        columns={"index": "household_id"}
    )

# 2. Check uniqueness before joining
assert households["household_id"].is_unique
assert beneficiaries["beneficiary_id"].is_unique

for table in [renewals, outreach, administrative, outcomes]:
    assert table["episode_id"].is_unique

assert fairness["beneficiary_id"].is_unique

# 3. Merge beneficiary and household information
data = renewals.merge(
    beneficiaries,
    on="beneficiary_id",
    how="left",
    validate="one_to_one"
)

data = data.merge(
    households,
    on="household_id",
    how="left",
    validate="many_to_one"
)

# 4. Merge outreach and administrative information
data = data.merge(
    outreach.drop(columns=["beneficiary_id"]),
    on="episode_id",
    how="left",
    validate="one_to_one"
)

data = data.merge(
    administrative.drop(columns=["beneficiary_id"]),
    on="episode_id",
    how="left",
    validate="one_to_one"
)

# 5. Add outcome (target only)
data = data.merge(
    outcomes[["episode_id", "churn_30d"]],
    on="episode_id",
    how="left",
    validate="one_to_one"
)

# 6. Define prediction-time columns
model_columns = [
    "episode_id",
    "beneficiary_id",
    "household_id",
    "prediction_date",
    "renewal_due_date",
    "age_group",
    "eligibility_category",
    "household_size",
    "income_to_fpl",
    "income_change_pct",
    "contact_stability",
    "address_stability",
    "months_covered",
    "prior_coverage_gaps",
    "previous_renewals",
    "notice_sent",
    "notice_returned",
    "outreach_attempts_pre_prediction",
    "responded_pre_prediction",
    "prior_documentation_issues",
    "income_verification_required",
    "administrative_complexity",
    "churn_30d"
]

model_data = data[model_columns].copy()

# 7. Build separate fairness audit table
audit_data = outcomes[
    ["episode_id", "beneficiary_id", "churn_30d"]
].merge(
    fairness,
    on="beneficiary_id",
    how="left",
    validate="one_to_one"
)

# 8. Validate
assert len(model_data) == 30000
assert len(audit_data) == 30000
assert model_data["episode_id"].is_unique
assert audit_data["episode_id"].is_unique
assert model_data.isna().sum().sum() == 0
assert audit_data.isna().sum().sum() == 0
assert model_data["churn_30d"].isin([0, 1]).all()

# Verify target matches between datasets
assert model_data.set_index("episode_id")[
    "churn_30d"
].sort_index().equals(
    audit_data.set_index("episode_id")[
        "churn_30d"
    ].sort_index()
)

# Verify prediction is exactly 30 days before renewal
prediction_dates = pd.to_datetime(model_data["prediction_date"])
renewal_dates = pd.to_datetime(model_data["renewal_due_date"])

assert (
    (renewal_dates - prediction_dates).dt.days == 30
).all()

# 9. Save final datasets
model_path = FINAL_DIR / "medicaid_modeling_dataset.csv"
audit_path = FINAL_DIR / "medicaid_fairness_audit.csv"

model_data.to_csv(model_path, index=False)
audit_data.to_csv(audit_path, index=False)

# 10. Report results
print("\nFinal modeling dataset created successfully!")
print("Modeling rows:", len(model_data))
print("Modeling columns:", len(model_data.columns))
print("Fairness audit rows:", len(audit_data))
print("Fairness audit columns:", len(audit_data.columns))

print("\nChurn distribution:")
print(model_data["churn_30d"].value_counts())

print("\nMissing values:", model_data.isna().sum().sum())
print("Duplicate episode IDs:", model_data["episode_id"].duplicated().sum())

print("\nSaved files:")
print(model_path)
print(audit_path)

print("\nFINAL DATASET VALIDATION: PASSED")
