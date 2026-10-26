
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Synthetic Medicaid Beneficiary Data Generator
# Project: Medicaid Enrollment Churn Decision-Support Tool
# ============================================================

# Set random seed for reproducibility
SEED = 20260930
rng = np.random.default_rng(SEED)

# Define input and output folder
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD HOUSEHOLD ASSIGNMENTS
# ============================================================

assignments = pd.read_csv(
    RAW_DIR / "beneficiary_household_assignments.csv"
)

n_beneficiaries = len(assignments)

print("Total assigned beneficiaries:", n_beneficiaries)

# ============================================================
# 2. GENERATE AGE GROUPS
# ============================================================

age_groups = ["0-18", "19-25", "26-44", "45-64", "65+"]

age_probabilities = [0.38, 0.10, 0.25, 0.20, 0.07]

age_group = rng.choice(
    age_groups,
    size=n_beneficiaries,
    p=age_probabilities
)

# ============================================================
# 3. GENERATE MEDICAID ELIGIBILITY CATEGORIES
# ============================================================

# Eligibility category probabilities depend on age group.
# These are simulation assumptions, not actual agency rates.

eligibility_configs = {
    "0-18": (
        ["child", "aged_blind_disabled", "other_special"],
        [0.91, 0.04, 0.05]
    ),

    "19-25": (
        [
            "adult_expansion",
            "parent_caretaker",
            "pregnancy_related",
            "aged_blind_disabled",
            "other_special"
        ],
        [0.56, 0.17, 0.10, 0.09, 0.08]
    ),

    "26-44": (
        [
            "adult_expansion",
            "parent_caretaker",
            "pregnancy_related",
            "aged_blind_disabled",
            "other_special"
        ],
        [0.48, 0.25, 0.10, 0.10, 0.07]
    ),

    "45-64": (
        [
            "adult_expansion",
            "parent_caretaker",
            "aged_blind_disabled",
            "other_special"
        ],
        [0.47, 0.12, 0.30, 0.11]
    ),

    "65+": (
        ["aged_blind_disabled", "other_special"],
        [0.82, 0.18]
    )
}

eligibility_category = np.empty(
    n_beneficiaries,
    dtype=object
)

for group, (categories, probabilities) in eligibility_configs.items():

    indices = np.where(age_group == group)[0]

    eligibility_category[indices] = rng.choice(
        categories,
        size=len(indices),
        p=probabilities
    )

# ============================================================
# 4. CREATE BENEFICIARY DATASET
# ============================================================

beneficiaries = assignments.copy()

beneficiaries["age_group"] = age_group

beneficiaries["eligibility_category"] = eligibility_category

# ============================================================
# 5. VALIDATE THE DATASET
# ============================================================

households = pd.read_csv(
    RAW_DIR / "01_households.csv"
)

# Support correction of an older household CSV
if "index" in households.columns:
    households = households.rename(
        columns={"index": "household_id"}
    )

assert len(beneficiaries) == 30000, \
    "Expected 30,000 beneficiaries"

assert beneficiaries["beneficiary_id"].is_unique, \
    "Duplicate beneficiary IDs found"

assert beneficiaries["household_id"].isin(
    households["household_id"]
).all(), "Some household IDs are missing"

assert beneficiaries.isnull().sum().sum() == 0, \
    "Missing values detected"

print("\nAll beneficiary validation checks passed!")

# ============================================================
# 6. SAVE THE DATASET
# ============================================================

output_file = RAW_DIR / "02_beneficiaries.csv"

beneficiaries.to_csv(
    output_file,
    index=False
)

print("\nBeneficiary dataset generated successfully!")
print("Saved to:", output_file)

print("\nNumber of beneficiaries:", len(beneficiaries))
print("Number of columns:", len(beneficiaries.columns))

print("\nFirst five rows:")
print(beneficiaries.head().to_string(index=False))

print("\nMissing values:")
print(beneficiaries.isnull().sum())

print("\nAge group distribution:")
print(beneficiaries["age_group"].value_counts())

print("\nEligibility category distribution:")
print(beneficiaries["eligibility_category"].value_counts())

print("\nHousehold relationship check: PASSED")
