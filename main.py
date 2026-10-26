from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MedChurn Insights",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/final/medicaid_fairness_audit.csv"


@st.cache_data
def episode_count(path, modified_time):
    return len(pd.read_csv(path, usecols=["episode_id"]))


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #F5F8FA;
        border-right: 1px solid #DFE7ED;
    }

    .sidebar-brand {
        padding: 18px 0 20px;
    }

    .sidebar-brand h2 {
        margin: 0;
        color: #14354B;
        font-size: 26px;
        font-weight: 750;
        letter-spacing: -0.5px;
    }

    .sidebar-brand p {
        margin: 6px 0 0;
        color: #607789;
        font-size: 13px;
    }

    .dataset-card {
        background: #FFFFFF;
        border: 1px solid #DFE7ED;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 22px;
    }

    .dataset-card .card-title {
        color: #607789;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin-bottom: 10px;
    }

    .dataset-card .episode-count {
        color: #14354B;
        font-size: 29px;
        font-weight: 750;
        line-height: 1.2;
    }

    .dataset-card .episode-label {
        color: #607789;
        font-size: 13px;
        margin-top: 4px;
    }

    .dataset-card .badge {
        display: inline-block;
        color: #176B70;
        background: #E8F4F2;
        border-radius: 6px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 12px;
    }

    .sidebar-section {
        color: #607789;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        margin: 16px 0 8px;
    }

    .sidebar-footer {
        border-top: 1px solid #DFE7ED;
        margin-top: 26px;
        padding-top: 14px;
        color: #607789;
        font-size: 12px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

overview = st.Page(
    "Renewal_Overview.py",
    title="Renewal Overview",
    icon=":material/dashboard:",
    default=True,
)

subgroups = st.Page(
    "pages/2_Outreach_and_Case_Review.py",
    title="Subgroup Analysis",
    icon=":material/groups:",
)

scenarios = st.Page(
    "pages/3_Predictive_Scenarios.py",
    title="Predictive Scenarios",
    icon=":material/query_stats:",
)

navigation = st.navigation(
    [overview, subgroups, scenarios],
    position="hidden",
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>MedChurn Insights</h2>
            <p>Medicaid renewal analytics</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if DATA_PATH.exists():
        count = episode_count(
            str(DATA_PATH),
            DATA_PATH.stat().st_mtime_ns,
        )
        count_text = f"{count:,}"
    else:
        count_text = "Unavailable"

    st.markdown(
        f"""
        <div class="dataset-card">
            <div class="card-title">DATASET INFORMATION</div>
            <div class="episode-count">{count_text}</div>
            <div class="episode-label">Renewal episodes</div>
            <span class="badge">Synthetic educational data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">EXPLORE</div>',
        unsafe_allow_html=True,
    )

    st.page_link(overview)
    st.page_link(subgroups)

    st.markdown(
        '<div class="sidebar-section">PREDICT & COMPARE</div>',
        unsafe_allow_html=True,
    )

    st.page_link(scenarios)

    st.markdown(
        """
        <div class="sidebar-footer">
            Supports renewal review and outreach planning.<br>
            Predictions do not determine eligibility.
        </div>
        """,
        unsafe_allow_html=True,
    )

navigation.run()