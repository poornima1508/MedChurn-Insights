
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Synthetic Medicaid Fairness Attributes Generator
# ============================================================

SEED = 20260930
rng = np.random.default_rng(SEED + 2)

RAW_DIR = Path("data/raw")

# Load beneficiaries
beneficiaries = pd.read_csv(
    RAW_DIR / "02_beneficiaries.csv"
)

n = len(beneficiaries)

# ============================================================
# 1. GENERATE RACE / ETHNICITY
# ============================================================

race_ethnicity = rng.choice(
    [
        "White",
        "Black",
        "Hispanic",
        "Asian",
        "Multiracial",
        "Other_or_Unknown"
    ],
    size=n,
    p=[0.49, 0.25, 0.13, 0.05, 0.04, 0.04]
)

# ============================================================
# 2. GENERATE PREFERRED LANGUAGE
# ============================================================

preferred_language = rng.choice(
    ["English", "Spanish", "Arabic", "Other"],
    size=n,
    p=[0.87, 0.08, 0.02, 0.03]
)

# ============================================================
# 3. GENERATE DISABILITY INDICATOR
# ============================================================

eligibility = beneficiaries[
    "eligibility_category"
].to_numpy()

age_group = beneficiaries["age_group"].to_numpy()

disability_probability = np.where(
    eligibility == "aged_blind_disabled",
    0.78,
    np.where(age_group == "65+", 0.20, 0.07)
)

disability_flag = rng.binomial(
    1,
    disability_probability
)

# ============================================================
# 4. CREATE FAIRNESS DATASET
# ============================================================

fairness = pd.DataFrame({
    "beneficiary_id": beneficiaries["beneficiary_id"],
    "race_ethnicity": race_ethnicity,
    "preferred_language": preferred_language,
    "disability_flag": np.where(
        disability_flag == 1, "Yes", "No"
    )
})

# ============================================================
# 5. VALIDATE
# ============================================================

assert len(fairness) == len(beneficiaries)
assert fairness["beneficiary_id"].is_unique
assert fairness.isnull().sum().sum() == 0

assert fairness["beneficiary_id"].equals(
    beneficiaries["beneficiary_id"]
)

# ============================================================
# 6. SAVE
# ============================================================

output_file = RAW_DIR / "03_fairness_attributes.csv"

fairness.to_csv(output_file, index=False)

print("Fairness dataset generated successfully!")
print("Number of rows:", len(fairness))
print("Number of columns:", len(fairness.columns))

print("\nFirst five rows:")
print(fairness.head().to_string(index=False))

print("\nRace/Ethnicity distribution:")
print(fairness["race_ethnicity"].value_counts())

print("\nLanguage distribution:")
print(fairness["preferred_language"].value_counts())

print("\nDisability distribution:")
print(fairness["disability_flag"].value_counts())

print("\nMissing values:")
print(fairness.isnull().sum())

print("\nFairness validation: PASSED")
