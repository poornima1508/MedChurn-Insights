
from pathlib import Path
import pandas as pd

# ============================================================
# STEP 9: COMPREHENSIVE FINAL DATASET VALIDATION
# ============================================================

FINAL_DIR = Path("data/final")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

model = pd.read_csv(
    FINAL_DIR / "medicaid_modeling_dataset.csv",
    parse_dates=["prediction_date", "renewal_due_date"]
)

audit = pd.read_csv(
    FINAL_DIR / "medicaid_fairness_audit.csv"
)

checks = []

def check(name, condition):
    passed = bool(condition)
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL"
    })
    print(f"{'PASS' if passed else 'FAIL'}: {name}")

# ============================================================
# 1. STRUCTURAL VALIDATION
# ============================================================

check("Modeling row count", len(model) == 30000)
check("Modeling column count", len(model.columns) == 23)
check("Audit row count", len(audit) == 30000)
check("Audit column count", len(audit.columns) == 6)

check("Unique modeling episode IDs", model["episode_id"].is_unique)
check("Unique audit episode IDs", audit["episode_id"].is_unique)
check("Unique beneficiary IDs", model["beneficiary_id"].is_unique)

# ============================================================
# 2. COMPLETENESS
# ============================================================

check("No missing modeling values", model.notna().all().all())
check("No missing audit values", audit.notna().all().all())

# ============================================================
# 3. TARGET VALIDATION
# ============================================================

check(
    "Binary churn target",
    model["churn_30d"].isin([0, 1]).all()
)

check(
    "Both target classes present",
    model["churn_30d"].nunique() == 2
)

check(
    "Churn target matches audit",
    model.set_index("episode_id")["churn_30d"]
    .sort_index()
    .equals(
        audit.set_index("episode_id")["churn_30d"]
        .sort_index()
    )
)

# ============================================================
# 4. TEMPORAL VALIDATION
# ============================================================

check(
    "Prediction occurs 30 days before renewal",
    (
        model["renewal_due_date"] -
        model["prediction_date"]
    ).dt.days.eq(30).all()
)

# ============================================================
# 5. DOMAIN VALIDATION
# ============================================================

check(
    "Valid household sizes",
    model["household_size"].between(1, 6).all()
)

check(
    "Nonnegative prior coverage gaps",
    model["prior_coverage_gaps"].ge(0).all()
)

check(
    "Nonnegative previous renewals",
    model["previous_renewals"].ge(0).all()
)

check(
    "Valid outreach attempt counts",
    model["outreach_attempts_pre_prediction"]
    .between(0, 3).all()
)

check(
    "Valid administrative complexity",
    model["administrative_complexity"]
    .isin(["Low", "Medium", "High"]).all()
)

check(
    "Valid binary administrative flags",
    model["prior_documentation_issues"].isin([0, 1]).all()
    and model["income_verification_required"].isin([0, 1]).all()
)

check(
    "Valid notice flags",
    model["notice_sent"].isin([0, 1]).all()
    and model["notice_returned"].isin([0, 1]).all()
)

check(
    "Returned notice requires sent notice",
    (model["notice_returned"] <= model["notice_sent"]).all()
)

check(
    "Response requires outreach attempt",
    (
        model["responded_pre_prediction"] <=
        (model["outreach_attempts_pre_prediction"] > 0)
    ).all()
)

# ============================================================
# 6. FAIRNESS ATTRIBUTE SEPARATION
# ============================================================

protected_columns = [
    "race_ethnicity",
    "preferred_language",
    "disability_flag"
]

check(
    "Fairness attributes excluded from modeling table",
    all(col not in model.columns for col in protected_columns)
)

check(
    "Fairness attributes present in audit table",
    all(col in audit.columns for col in protected_columns)
)

# ============================================================
# 7. SAVE VALIDATION RESULTS
# ============================================================

results = pd.DataFrame(checks)

output_path = DOCS_DIR / "validation_results.csv"
results.to_csv(output_path, index=False)

passed = (results["status"] == "PASS").sum()
failed = (results["status"] == "FAIL").sum()

print("\n===================================")
print("VALIDATION SUMMARY")
print("===================================")
print("Total checks:", len(results))
print("Passed:", passed)
print("Failed:", failed)
print("Churn rate:", f"{model['churn_30d'].mean():.2%}")
print("Results saved to:", output_path)

if failed:
    raise SystemExit("VALIDATION FAILED: review the failed checks.")

print("\nCOMPREHENSIVE VALIDATION: PASSED")
