
from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# Synthetic Medicaid Renewal Notices and Outreach Generator
# ============================================================

SEED = 20260930
rng = np.random.default_rng(SEED + 4)

RAW_DIR = Path("data/raw")

renewals = pd.read_csv(
    RAW_DIR / "04_renewal_episodes.csv",
    parse_dates=["renewal_due_date", "prediction_date"]
)

households = pd.read_csv(
    RAW_DIR / "01_households.csv"
)

beneficiaries = pd.read_csv(
    RAW_DIR / "02_beneficiaries.csv"
)

# Handle an older household file if needed
if "index" in households.columns:
    households = households.rename(
        columns={"index": "household_id"}
    )

# Join household stability indicators
data = renewals.merge(
    beneficiaries[["beneficiary_id", "household_id"]],
    on="beneficiary_id",
    how="left",
    validate="one_to_one"
).merge(
    households[
        ["household_id", "contact_stability", "address_stability"]
    ],
    on="household_id",
    how="left",
    validate="many_to_one"
)

assert len(data) == 30000
assert data[
    ["contact_stability", "address_stability"]
].notna().all().all()

n = len(data)

# ============================================================
# 1. SIMULATE NOTICES AVAILABLE BEFORE PREDICTION
# ============================================================

# Initial notice occurs 45 days before renewal.
notice_sent_date = (
    data["renewal_due_date"] - pd.Timedelta(days=45)
)

notice_sent = rng.binomial(1, 0.97, n)

# Returned-mail probability varies by address stability.
return_probabilities = {
    "High": 0.03,
    "Medium": 0.11,
    "Low": 0.27
}

return_probability = data[
    "address_stability"
].map(return_probabilities).to_numpy()

assert not np.isnan(return_probability).any()

notice_returned = (
    rng.binomial(1, return_probability) * notice_sent
)

# ============================================================
# 2. SIMULATE EARLY OUTREACH
# ============================================================

# Attempts are limited to the period between the notice
# date and the prediction date (15 days).

outreach_attempts = rng.choice(
    [0, 1, 2, 3],
    size=n,
    p=[0.30, 0.39, 0.23, 0.08]
)

response_probabilities = {
    "High": 0.72,
    "Medium": 0.49,
    "Low": 0.25
}

response_probability = data[
    "contact_stability"
].map(response_probabilities).to_numpy()

assert not np.isnan(response_probability).any()

responded_before_prediction = rng.binomial(
    1, response_probability
)

# A response requires at least one outreach attempt.
responded_before_prediction = (
    responded_before_prediction *
    (outreach_attempts > 0)
)

# ============================================================
# 3. BUILD OUTREACH DATASET
# ============================================================

outreach = pd.DataFrame({
    "episode_id": data["episode_id"],
    "beneficiary_id": data["beneficiary_id"],
    "notice_sent_date": notice_sent_date,
    "notice_sent": notice_sent,
    "notice_returned": notice_returned,
    "outreach_attempts_pre_prediction": outreach_attempts,
    "responded_pre_prediction": responded_before_prediction
})

# No notice date should occur after prediction.
assert (
    outreach["notice_sent_date"] <=
    data["prediction_date"]
).all()

# ============================================================
# 4. VALIDATE
# ============================================================

assert len(outreach) == 30000
assert outreach["episode_id"].is_unique
assert outreach["beneficiary_id"].is_unique
assert outreach.isnull().sum().sum() == 0

assert (
    outreach["responded_pre_prediction"] <=
    (outreach["outreach_attempts_pre_prediction"] > 0)
).all()

assert (
    outreach["notice_returned"] <=
    outreach["notice_sent"]
).all()

# ============================================================
# 5. SAVE
# ============================================================

output_file = RAW_DIR / "05_renewal_outreach.csv"

outreach.to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)

print("Renewal outreach dataset generated successfully!")
print("Number of records:", len(outreach))
print("Number of columns:", len(outreach.columns))

print("\nFirst five rows:")
print(outreach.head().to_string(index=False))

print("\nNotices sent:")
print(outreach["notice_sent"].value_counts())

print("\nNotices returned:")
print(outreach["notice_returned"].value_counts())

print("\nResponses before prediction:")
print(outreach["responded_pre_prediction"].value_counts())

print("\nMissing values:")
print(outreach.isnull().sum())

print("\nOutreach validation: PASSED")
