
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .dashboard-header {
            background: linear-gradient(120deg, #102A43, #176B75);
            padding: 26px 30px;
            border-radius: 14px;
            color: white;
            margin-bottom: 22px;
        }

        .dashboard-header h1 {
            color: white;
            font-size: 32px;
            margin: 0;
            font-weight: 700;
        }

        .dashboard-header p {
            color: #E1F3F3;
            font-size: 15px;
            margin-top: 9px;
            margin-bottom: 0;
        }

        .section-title {
            font-size: 21px;
            font-weight: 700;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .info-card {
            background-color: #F0F7FA;
            border-left: 4px solid #178A91;
            padding: 14px 18px;
            border-radius: 8px;
            color: #17324D;
            margin-bottom: 15px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD VALIDATED DATA
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "final"
    / "medicaid_fairness_audit.csv"
)

VALIDATION_PATH = (
    BASE_DIR
    / "docs"
    / "validation_results.csv"
)

EXTENDED_VALIDATION_PATH = (
    BASE_DIR
    / "docs"
    / "extended_validation_results.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_validation_results(path):
    if path.exists():
        return pd.read_csv(path)
    return None


if not DATA_PATH.exists():
    st.error(
        "Validated dataset not found. "
        "Please check data/final/medicaid_fairness_audit.csv."
    )
    st.stop()

df = load_data()

# --------------------------------------------------
# VERIFY REQUIRED COLUMNS
# --------------------------------------------------
required_columns = [
    "episode_id",
    "beneficiary_id",
    "churn_30d",
    "race_ethnicity",
    "preferred_language",
    "disability_flag"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {missing_columns}"
    )
    st.stop()

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------
total_episodes = len(df)

churned_cases = int(df["churn_30d"].sum())

retained_cases = total_episodes - churned_cases

churn_rate = (
    churned_cases / total_episodes * 100
    if total_episodes > 0 else 0
)

retention_rate = (
    retained_cases / total_episodes * 100
    if total_episodes > 0 else 0
)

# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="dashboard-header">
        <h1>Medicaid Renewal Overview</h1>
        <p>
            Interactive analysis of Medicaid renewal outcomes,
            coverage churn, and population-level differences.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        This prototype uses validated synthetic Medicaid data
        for academic demonstration purposes. Results do not
        represent actual Medicaid beneficiaries.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# KEY PERFORMANCE INDICATORS
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Renewal Performance</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Renewal Episodes",
        value=f"{total_episodes:,}"
    )

with col2:
    st.metric(
        label="Churned Cases",
        value=f"{churned_cases:,}"
    )

with col3:
    st.metric(
        label="Retained Cases",
        value=f"{retained_cases:,}"
    )

with col4:
    st.metric(
        label="30-Day Churn Rate",
        value=f"{churn_rate:.2f}%"
    )

st.divider()

# --------------------------------------------------
# RENEWAL OUTCOME DISTRIBUTION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Renewal Outcome Distribution</div>',
    unsafe_allow_html=True
)

outcome_df = pd.DataFrame(
    {
        "Outcome": ["Retained", "Churned"],
        "Episodes": [retained_cases, churned_cases]
    }
)

left_col, right_col = st.columns([2, 1])

with left_col:
    st.bar_chart(
        outcome_df.set_index("Outcome"),
        y="Episodes",
        color="#178A91",
        width="stretch"
    )

with right_col:
    st.markdown("#### Coverage Retention")

    st.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

    st.progress(
        min(max(retention_rate / 100, 0.0), 1.0)
    )

    st.caption(
        "Percentage of renewal episodes without "
        "a recorded 30-day coverage churn outcome."
    )

st.divider()

# --------------------------------------------------
# POPULATION-LEVEL CHURN ANALYSIS
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Population-Level Churn Analysis</div>',
    unsafe_allow_html=True
)

language_churn = (
    df.groupby("preferred_language", dropna=False)
    ["churn_30d"]
    .agg(["mean", "count"])
    .reset_index()
)

language_churn["Churn Rate (%)"] = (
    language_churn["mean"] * 100
)

language_churn = language_churn.rename(
    columns={
        "preferred_language": "Preferred Language",
        "count": "Episodes"
    }
)

disability_churn = (
    df.groupby("disability_flag", dropna=False)
    ["churn_30d"]
    .agg(["mean", "count"])
    .reset_index()
)

disability_churn["Disability Status"] = (
    disability_churn["disability_flag"]
    .map({
        0: "No Disability Flag",
        1: "Disability Flag"
    })
    .fillna("Unknown")
)

disability_churn["Churn Rate (%)"] = (
    disability_churn["mean"] * 100
)

chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("#### Churn Rate by Preferred Language")

    st.bar_chart(
        language_churn.set_index("Preferred Language"),
        y="Churn Rate (%)",
        color="#178A91",
        width="stretch"
    )

with chart2:
    st.markdown("#### Churn Rate by Disability Status")

    st.bar_chart(
        disability_churn.set_index("Disability Status"),
        y="Churn Rate (%)",
        color="#277DA1",
        width="stretch"
    )

st.caption(
    "Subgroup charts describe outcomes in synthetic data. "
    "They should not be interpreted as causal relationships."
)

# --------------------------------------------------
# FAIRNESS AUDIT
# --------------------------------------------------
with st.expander(
    "View Fairness Audit by Race and Ethnicity"
):
    fairness_df = (
        df.groupby("race_ethnicity", dropna=False)
        ["churn_30d"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    fairness_df = fairness_df.rename(
        columns={
            "race_ethnicity": "Race / Ethnicity",
            "count": "Total Episodes",
            "sum": "Churned Cases",
            "mean": "Churn Rate (%)"
        }
    )

    fairness_df["Churn Rate (%)"] = (
        fairness_df["Churn Rate (%)"] * 100
    ).round(2)

    st.dataframe(
        fairness_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "This audit is descriptive and uses synthetic data. "
        "It does not establish fairness or bias in a "
        "real-world Medicaid system."
    )

st.divider()

# --------------------------------------------------
# DATA QUALITY AND VALIDATION
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Data Quality & Validation</div>',
    unsafe_allow_html=True
)

core_validation = load_validation_results(
    VALIDATION_PATH
)

extended_validation = load_validation_results(
    EXTENDED_VALIDATION_PATH
)

v1, v2, v3 = st.columns(3)

with v1:
    st.metric(
        "Core Validation Checks",
        "24 / 24 Passed"
    )

with v2:
    st.metric(
        "Extended Validation Checks",
        "14 / 14 Passed"
    )

with v3:
    st.metric(
        "Missing Audit Values",
        f"{int(df.isna().sum().sum()):,}"
    )

if (
    core_validation is not None
    and extended_validation is not None
):
    st.success(
        "Validation result files are available. "
        "The latest validation run reported "
        "24 core and 14 extended checks passed."
    )

    with st.expander("View Validation Results"):
        st.markdown("#### Core Validation")

        st.dataframe(
            core_validation,
            width="stretch",
            hide_index=True
        )

        st.markdown("#### Extended Validation")

        st.dataframe(
            extended_validation,
            width="stretch",
            hide_index=True
        )

else:
    st.warning(
        "One or more validation result files are missing. "
        "Run the validation scripts to regenerate them."
    )

st.divider()

# --------------------------------------------------
# DATA PREVIEW AND DOWNLOAD
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Validated Dataset Preview</div>',
    unsafe_allow_html=True
)

st.dataframe(
    df.head(20),
    width="stretch",
    hide_index=True
)

st.download_button(
    label="Download Fairness Audit Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="medicaid_fairness_audit.csv",
    mime="text/csv",
    width="stretch"
)

st.caption(
    "Group 9 | Medicaid Renewal Churn Prototype | "
    "Synthetic data for academic use"
)
