
from pathlib import Path
import pandas as pd

FINAL_DIR = Path("data/final")
DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

model = pd.read_csv(
    FINAL_DIR / "medicaid_modeling_dataset.csv"
)

audit = pd.read_csv(
    FINAL_DIR / "medicaid_fairness_audit.csv"
)

descriptions = {
    "episode_id": "Unique synthetic renewal episode identifier",
    "beneficiary_id": "Unique synthetic beneficiary identifier",
    "household_id": "Identifier linking a beneficiary to a household",
    "prediction_date": "Date when churn risk is predicted",
    "renewal_due_date": "Scheduled Medicaid renewal date",
    "age_group": "Simulated beneficiary age category",
    "eligibility_category": "Simulated Medicaid eligibility category",
    "household_size": "Number of beneficiaries assigned to the household",
    "income_to_fpl": "Simulated household income relative to the federal poverty level",
    "income_change_pct": "Simulated proportional household income change",
    "contact_stability": "Simulated reliability of beneficiary contact information",
    "address_stability": "Simulated stability of household address",
    "months_covered": "Approximate months of prior coverage at prediction",
    "prior_coverage_gaps": "Number of historical coverage interruptions",
    "previous_renewals": "Simulated count of previous renewal events",
    "notice_sent": "Whether an initial renewal notice was sent",
    "notice_returned": "Whether a sent renewal notice was returned",
    "outreach_attempts_pre_prediction": "Number of outreach attempts before prediction",
    "responded_pre_prediction": "Whether beneficiary responded before prediction",
    "prior_documentation_issues": "Indicator of previous documentation problems",
    "income_verification_required": "Simulated additional income verification requirement",
    "administrative_complexity": "Derived administrative complexity category",
    "churn_30d": "Simulated coverage interruption within 30 days after renewal",
    "race_ethnicity": "Synthetic race/ethnicity category for fairness auditing",
    "preferred_language": "Synthetic preferred language for fairness auditing",
    "disability_flag": "Synthetic disability indicator for fairness auditing"
}

identifier_columns = {
    "episode_id", "beneficiary_id", "household_id"
}

date_columns = {
    "prediction_date", "renewal_due_date"
}

audit_only_columns = {
    "race_ethnicity", "preferred_language", "disability_flag"
}

def get_role(column):
    if column == "churn_30d":
        return "Target"
    if column in audit_only_columns:
        return "Fairness audit only"
    if column in identifier_columns:
        return "Identifier - exclude from model"
    if column in date_columns:
        return "Temporal metadata - exclude from initial model"
    return "Candidate predictor"

rows = []

for dataset_name, df in [
    ("Modeling", model),
    ("Fairness Audit", audit)
]:
    for column in df.columns:
        rows.append({
            "dataset": dataset_name,
            "column": column,
            "data_type": str(df[column].dtype),
            "description": descriptions.get(
                column, "Description required"
            ),
            "role": get_role(column),
            "missing_count": int(df[column].isna().sum()),
            "unique_values": int(df[column].nunique())
        })

dictionary = pd.DataFrame(rows)

output_file = DOCS_DIR / "data_dictionary.csv"
dictionary.to_csv(output_file, index=False)

print("\nData dictionary created successfully!")
print("Total dictionary entries:", len(dictionary))
print("Missing descriptions:",
      (dictionary["description"] == "Description required").sum())
print("Saved to:", output_file)

print("\nDataset breakdown:")
print(dictionary["dataset"].value_counts())

print("\nDATA DICTIONARY: COMPLETE")
