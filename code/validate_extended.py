
from pathlib import Path
import pandas as pd

raw = Path("data/raw")
final = Path("data/final")
docs = Path("docs")
docs.mkdir(exist_ok=True)

model = pd.read_csv(final / "medicaid_modeling_dataset.csv")
audit = pd.read_csv(final / "medicaid_fairness_audit.csv")
households = pd.read_csv(raw / "01_households.csv")
beneficiaries = pd.read_csv(raw / "02_beneficiaries.csv")
renewals = pd.read_csv(raw / "04_renewal_episodes.csv")
outcomes = pd.read_csv(raw / "07_churn_outcomes.csv")

if "index" in households.columns:
    households = households.rename(
        columns={"index": "household_id"}
    )

results = []

def check(name, condition):
    status = "PASS" if bool(condition) else "FAIL"
    results.append({"check": name, "status": status})
    print(f"{status}: {name}")

# Data types
numeric_fields = [
    "household_size",
    "income_to_fpl",
    "income_change_pct",
    "months_covered",
    "prior_coverage_gaps",
    "previous_renewals",
    "outreach_attempts_pre_prediction",
    "churn_30d"
]

check(
    "Numeric feature types",
    all(pd.api.types.is_numeric_dtype(model[c])
        for c in numeric_fields)
)

check(
    "Valid prediction dates",
    pd.to_datetime(
        model["prediction_date"], errors="coerce"
    ).notna().all()
)

check(
    "Valid renewal dates",
    pd.to_datetime(
        model["renewal_due_date"], errors="coerce"
    ).notna().all()
)

# Referential integrity
check(
    "All household IDs exist",
    model["household_id"].isin(
        households["household_id"]
    ).all()
)

check(
    "All beneficiary IDs exist",
    model["beneficiary_id"].isin(
        beneficiaries["beneficiary_id"]
    ).all()
)

check(
    "All episode IDs exist",
    model["episode_id"].isin(
        renewals["episode_id"]
    ).all()
)

check(
    "All outcomes link to model episodes",
    model["episode_id"].isin(
        outcomes["episode_id"]
    ).all()
)

check(
    "Model and audit episode sets match",
    set(model["episode_id"]) ==
    set(audit["episode_id"])
)

# Check beneficiary-household assignment consistency
reference = beneficiaries.set_index(
    "beneficiary_id"
)["household_id"]

check(
    "Beneficiary household assignments match",
    model["household_id"].equals(
        model["beneficiary_id"].map(reference)
    )
)

# Derived complexity reconstruction
complexity_score = (
    model["prior_documentation_issues"]
    + model["income_verification_required"]
    + (model["prior_coverage_gaps"] > 0).astype(int)
)

expected_complexity = complexity_score.map(
    {0: "Low", 1: "Medium", 2: "High", 3: "High"}
)

check(
    "Administrative complexity is reproducible",
    expected_complexity.equals(
        model["administrative_complexity"]
    )
)

# Basic metric computability
check(
    "Churn rate can be computed",
    0 <= model["churn_30d"].mean() <= 1
)

check(
    "Churn counts reconcile",
    model["churn_30d"].value_counts().sum()
    == len(model)
)

joined = audit.merge(
    model[["episode_id", "churn_30d"]],
    on="episode_id",
    suffixes=("_audit", "_model"),
    validate="one_to_one"
)

check(
    "Audit and modeling outcomes agree",
    (
        joined["churn_30d_audit"]
        == joined["churn_30d_model"]
    ).all()
)

check(
    "Subgroup churn rates can be computed",
    all(
        audit.groupby(column)["churn_30d"]
        .mean().notna().all()
        for column in [
            "race_ethnicity",
            "preferred_language",
            "disability_flag"
        ]
    )
)

output = docs / "extended_validation_results.csv"
pd.DataFrame(results).to_csv(output, index=False)

passed = sum(r["status"] == "PASS" for r in results)
failed = len(results) - passed

print("\nEXTENDED VALIDATION SUMMARY")
print("Total checks:", len(results))
print("Passed:", passed)
print("Failed:", failed)
print("Saved to:", output)

if failed:
    raise SystemExit("Extended validation failed.")

print("\nEXTENDED VALIDATION: PASSED")
