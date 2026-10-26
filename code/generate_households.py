
from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible random number generator
SEED = 20260930
rng = np.random.default_rng(SEED)

# Target number of synthetic Medicaid beneficiaries
N_BENEFICIARIES = 30000

# Output folder
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Household size assumptions
household_sizes = np.arange(1, 7)
household_probabilities = [
    0.28, 0.31, 0.20, 0.12, 0.06, 0.03
]

# Estimate the number of households
n_households = int(np.ceil(N_BENEFICIARIES / 1.75))

sizes = rng.choice(
    household_sizes,
    size=n_households,
    p=household_probabilities
)

# Add households until there are enough member slots
while sizes.sum() < N_BENEFICIARIES:
    extra = int(np.ceil(
        (N_BENEFICIARIES - sizes.sum()) / 1.75
    ))
    sizes = np.concatenate([
        sizes,
        rng.choice(
            household_sizes,
            size=extra,
            p=household_probabilities
        )
    ])

# Assign synthetic beneficiaries to household IDs
household_ids = [
    f"HH{i:06d}"
    for i in range(1, len(sizes) + 1)
]

household_slots = np.repeat(household_ids, sizes)
rng.shuffle(household_slots)

assigned_households = household_slots[:N_BENEFICIARIES]

# Count actual assigned members in each household
actual_sizes = (
    pd.Series(assigned_households)
    .value_counts()
    .sort_index()
)

# Generate household-level characteristics
household_base = pd.DataFrame({
    "household_id": household_ids,
    "income_to_fpl": np.round(
        np.clip(
            rng.lognormal(
                np.log(1.35), 0.42, len(sizes)
            ),
            0.2, 4.0
        ),
        3
    ),
    "income_change_pct": np.round(
        100 * np.clip(
            rng.normal(0, 0.20, len(sizes)),
            -0.75, 1.25
        ),
        1
    ),
    "contact_stability": rng.choice(
        ["High", "Medium", "Low"],
        len(sizes),
        p=[0.60, 0.28, 0.12]
    ),
    "address_stability": rng.choice(
        ["High", "Medium", "Low"],
        len(sizes),
        p=[0.72, 0.20, 0.08]
    )
}).set_index("household_id")

# Keep households with assigned beneficiaries
households = household_base.loc[
    actual_sizes.index
].copy()

households.insert(
    0, "household_size", actual_sizes.to_numpy()
)

households = households.reset_index().rename(
    columns={"index": "household_id"}
)

# Save the first dataset
output_file = OUTPUT_DIR / "01_households.csv"
households.to_csv(output_file, index=False)

# Save household assignments for the next step
pd.DataFrame({
    "beneficiary_id": [
        f"B{i:07d}"
        for i in range(1, N_BENEFICIARIES + 1)
    ],
    "household_id": assigned_households
}).to_csv(
    OUTPUT_DIR / "beneficiary_household_assignments.csv",
    index=False
)

print("Household dataset generated successfully!")
print("Number of households:", len(households))
print("Number of columns:", len(households.columns))
print("Assigned beneficiaries:", len(assigned_households))
print("\nFirst five rows:")
print(households.head())
print("\nMissing values:")
print(households.isnull().sum())
