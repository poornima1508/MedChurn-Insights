from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "final"
    / "medicaid_fairness_audit.csv"
)


# --------------------------------------------------
# LOAD AND CHECK DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required = [
        "episode_id",
        "beneficiary_id",
        "churn_30d",
        "preferred_language",
        "disability_flag",
        "race_ethnicity",
    ]

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if data[required].isna().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    data["churn_30d"] = pd.to_numeric(
        data["churn_30d"],
        errors="raise",
    )

    if not data["churn_30d"].isin([0, 1]).all():
        raise ValueError("Churn labels must be 0 or 1.")

    data["churn_30d"] = data["churn_30d"].astype(int)

    # Accept text and numeric representations.
    disability_values = (
        data["disability_flag"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    normalized_flags = disability_values.map(
        {
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
            "1.0": 1,
            "0.0": 0,
            "true": 1,
            "false": 0,
        }
    )

    if normalized_flags.isna().any():
        invalid_values = (
            disability_values[normalized_flags.isna()]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unrecognized disability flag values: "
            f"{invalid_values}"
        )

    data["disability_flag"] = normalized_flags.astype(int)

    if not data["episode_id"].is_unique:
        raise ValueError("Episode IDs must be unique.")

    return data


try:
    df = load_data()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Data loading failed: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div style="
        background:linear-gradient(110deg,#102E48,#176B82);
        padding:30px;
        border-radius:14px;
        margin-bottom:24px;
    ">
        <h1 style="
            color:white;
            font-size:34px;
            margin:0 0 12px;
        ">
            Subgroup Analysis
        </h1>
        <p style="
            color:#DCECF3;
            font-size:16px;
            margin:0;
        ">
            Explore historical synthetic renewal outcomes
            and review cases through interactive filters.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "This page describes historical synthetic outcomes. "
    "It does not predict future churn or evaluate model fairness."
)


# --------------------------------------------------
# FILTERS
# --------------------------------------------------
st.subheader("Explore Renewal Cases")
st.caption("Apply filters to update the case summary and table.")

c1, c2, c3 = st.columns(3)

with c1:
    outcome = st.selectbox(
        "Renewal Outcome",
        ["All", "Churned", "Retained"],
    )

with c2:
    languages = sorted(
        df["preferred_language"]
        .dropna()
        .unique()
        .tolist()
    )

    language = st.selectbox(
        "Preferred Language",
        ["All"] + languages,
    )

with c3:
    disability = st.selectbox(
        "Disability Status",
        ["All", "No Disability Flag", "Disability Flag"],
    )

search_id = st.text_input(
    "Search by Episode ID",
    placeholder="Example: R0000001",
)


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------
filtered = df.copy()

if outcome == "Churned":
    filtered = filtered[filtered["churn_30d"] == 1]
elif outcome == "Retained":
    filtered = filtered[filtered["churn_30d"] == 0]

if language != "All":
    filtered = filtered[
        filtered["preferred_language"] == language
    ]

if disability == "No Disability Flag":
    filtered = filtered[filtered["disability_flag"] == 0]
elif disability == "Disability Flag":
    filtered = filtered[filtered["disability_flag"] == 1]

if search_id.strip():
    filtered = filtered[
        filtered["episode_id"]
        .astype(str)
        .str.contains(
            search_id.strip(),
            case=False,
            regex=False,
        )
    ]


# --------------------------------------------------
# FILTERED SUMMARY
# --------------------------------------------------
total = len(filtered)
churned = int(filtered["churn_30d"].sum())
retained = total - churned
rate = churned / total * 100 if total else None

st.divider()
st.subheader("Filtered Renewal Summary")

k1, k2, k3, k4 = st.columns(4)

with k1:
    with st.container(border=True):
        st.metric("Matching Cases", f"{total:,}")

with k2:
    with st.container(border=True):
        st.metric("Churned Cases", f"{churned:,}")

with k3:
    with st.container(border=True):
        st.metric("Retained Cases", f"{retained:,}")

with k4:
    with st.container(border=True):
        st.metric(
            "Churn Rate",
            f"{rate:.2f}%" if rate is not None else "N/A",
        )

st.caption(
    "The churn rate uses only matching cases as its denominator. "
    "Filtering by renewal outcome restricts the result to that "
    "outcome, so its churn rate will be 100% or 0% when cases match."
)


# --------------------------------------------------
# CASE TABLE
# --------------------------------------------------
st.divider()
st.subheader("Historical Renewal Case Review")

review = filtered.copy()

review["Renewal Outcome"] = review["churn_30d"].map(
    {
        1: "Churned",
        0: "Retained",
    }
)

review["Disability Status"] = review["disability_flag"].map(
    {
        0: "No Disability Flag",
        1: "Disability Flag",
    }
)

display_columns = [
    "episode_id",
    "beneficiary_id",
    "Renewal Outcome",
    "preferred_language",
    "Disability Status",
    "race_ethnicity",
]

display = review[display_columns].rename(
    columns={
        "episode_id": "Episode ID",
        "beneficiary_id": "Beneficiary ID",
        "preferred_language": "Preferred Language",
        "race_ethnicity": "Race / Ethnicity",
    }
)

if display.empty:
    st.warning(
        "No cases match these filters. "
        "Adjust the filters or clear the episode search."
    )

st.dataframe(
    display,
    hide_index=True,
    width="stretch",
    height=400,
)


# --------------------------------------------------
# EXPORT
# --------------------------------------------------
st.download_button(
    label="Download Filtered Cases",
    data=display.to_csv(index=False).encode("utf-8"),
    file_name="medicaid_filtered_cases.csv",
    mime="text/csv",
    disabled=display.empty,
)


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------
st.divider()

st.info(
    "Renewal outcomes are simulated historical labels, "
    "not predictions of future churn. Use Predictive Scenarios "
    "to explore model-based risk estimates."
)

st.caption(
    "Population attributes support descriptive subgroup analysis. "
    "Differences do not establish causation, and these comparisons "
    "do not establish predictive-model fairness or justify "
    "eligibility decisions."
)

st.caption(
    "MedChurn | Synthetic Data | Educational Application"
)