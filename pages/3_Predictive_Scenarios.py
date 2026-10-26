import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from modeling import SPECS, explain, summarize, train_all


LABELS = {
    "prior_coverage_gaps": "Prior Coverage Gaps",
    "notice_sent": "Renewal Notice Sent",
    "notice_returned": "Renewal Notice Returned",
    "responded_pre_prediction": "Responded Before Prediction",
    "prior_documentation_issues": "Prior Documentation Issues",
    "administrative_complexity": "Administrative Complexity",
    "contact_stability": "Contact Stability",
    "address_stability": "Address Stability",
}


def label(column):
    return LABELS.get(column, column.replace("_", " ").title())


@st.cache_resource
def fitted():
    return train_all()


with st.spinner("Preparing predictive models..."):
    data, bundles = fitted()

st.title("Renewal Risk and Outreach Scenarios")

st.warning(
    "Synthetic educational data. Predictions support planning "
    "and do not establish causal effects or determine eligibility."
)

name = st.selectbox("Select a prediction model", list(SPECS))
spec = SPECS[name]
bundle = bundles[name]

st.caption("Prediction timeline: " + spec["time"])

if name == "Coverage churn":
    st.markdown(
        "**KPI — Coverage churn rate:** The proportion of renewal "
        "episodes with a simulated coverage interruption within "
        "30 days after renewal."
    )
    st.write(
        "The Renewal Operations Supervisor can use estimated risk "
        "to prioritize cases for coverage-continuity assistance."
    )
else:
    st.markdown(
        "**KPI — Early outreach nonresponse rate:** The proportion "
        "of renewal episodes without a response by 30 days before "
        "renewal, including episodes with no outreach attempts."
    )
    st.write(
        "The Renewal Operations Supervisor can use this earlier "
        "forecast to plan outreach and estimate unresolved workload. "
        "It assumes contact and address stability are known "
        "45 days before renewal."
    )

st.caption(
    "Individual probabilities can be averaged over an appropriate "
    "cohort to estimate its expected outcome rate. "
    "Neither model estimates financial savings."
)

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
with st.expander("Model performance and limitations"):
    st.json(bundle["metrics"])

    st.write(
        "Both models use logistic regression. Training and testing "
        "use separate households with a fixed approximately 75/25 "
        "split. Preprocessing is fitted only on training data."
    )
    st.write(
        "Logistic regression is appropriate for binary outcomes, "
        "produces probability estimates, and supports exact "
        "additive explanations in log odds."
    )
    st.write(
        "Lower Brier score and log loss indicate better probability "
        "predictions. Compare average precision with test prevalence. "
        "The holdout was not used for model tuning."
    )
    st.info(
        "These results are from an internal synthetic-data holdout. "
        "They do not establish performance on real Medicaid records "
        "or future renewal periods."
    )

st.divider()

# --------------------------------------------------
# OVERALL IMPORTANCE
# --------------------------------------------------
st.subheader("Overall Predictor Importance")

st.caption(
    "This measures predictive usefulness across held-out cases. "
    "Importance is the increase in log loss after shuffling a "
    "predictor, averaged over five repetitions. Larger positive "
    "values indicate greater usefulness. Near-zero or negative "
    "values provide little evidence of usefulness."
)

importance = bundle["importance"].copy()
importance["predictor"] = importance["predictor"].map(label)
importance = importance.reset_index(drop=True)

st.dataframe(
    importance.rename(
        columns={
            "predictor": "Predictor",
            "importance": "Mean Increase in Log Loss",
            "sd": "Standard Deviation",
        }
    ),
    hide_index=True,
    width="stretch",
)

importance_chart = (
    alt.Chart(importance)
    .mark_bar(color="#178A91")
    .encode(
        x=alt.X(
            "importance:Q",
            title="Mean Increase in Log Loss",
            scale=alt.Scale(zero=True),
        ),
        y=alt.Y(
            "predictor:N",
            title=None,
            sort=importance["predictor"].tolist(),
            axis=alt.Axis(labelLimit=300),
        ),
        tooltip=[
            alt.Tooltip("predictor:N", title="Predictor"),
            alt.Tooltip(
                "importance:Q",
                title="Mean Increase in Log Loss",
                format=".6f",
            ),
            alt.Tooltip(
                "sd:Q",
                title="Standard Deviation",
                format=".6f",
            ),
        ],
    )
    .properties(height=max(150, len(importance) * 40))
)

st.altair_chart(importance_chart, width="stretch")

st.info(
    "Overall importance is different from a contribution to one "
    "prediction. Neither measure demonstrates causation. "
    "Correlated predictors can share or mask importance."
)

st.divider()

# --------------------------------------------------
# BASELINE
# --------------------------------------------------
st.subheader("Baseline Case")

idx = st.selectbox(
    "Select a held-out renewal episode",
    bundle["test"][:100].tolist(),
    format_func=lambda i: str(data.loc[i, "episode_id"]),
    key=f"baseline_{name}",
)

baseline = data.loc[[idx], spec["features"]].copy()
baseline_probability = float(
    bundle["pipe"].predict_proba(baseline)[0, 1]
)

st.metric("Baseline Predicted Risk", f"{baseline_probability:.2%}")

st.dataframe(
    baseline.rename(columns=label),
    hide_index=True,
    width="stretch",
)

st.caption(
    "This selected episode is the baseline for your what-if "
    "comparison. Its prediction remains fixed while you modify "
    "the scenario inputs."
)

# --------------------------------------------------
# SCENARIO INPUTS
# --------------------------------------------------
with st.form(f"scenario_{name}_{idx}"):
    st.subheader("What-if Scenario")

    if name == "Coverage churn":
        st.caption(
            "Modify inputs and run the analysis. For binary fields, "
            "0 means No and 1 means Yes. Unchanged inputs are allowed."
        )
    else:
        st.caption(
            "Modify contact or address stability and run the "
            "analysis. Unchanged inputs are allowed."
        )

    values = {}

    for column in spec["features"]:
        original = baseline.iloc[0][column]
        widget_key = f"input_{name}_{idx}_{column}"

        if not pd.api.types.is_numeric_dtype(data[column]):
            options = sorted(
                data[column].dropna().unique().tolist()
            )
            values[column] = st.selectbox(
                label(column),
                options,
                index=options.index(original),
                key=widget_key,
            )
        else:
            values[column] = st.number_input(
                label(column),
                min_value=int(data[column].min()),
                max_value=int(data[column].max()),
                value=int(original),
                step=1,
                key=widget_key,
            )

    run = st.form_submit_button("Run What-if Analysis")

# --------------------------------------------------
# VALIDATE AND RETAIN LAST SUBMITTED SCENARIO
# --------------------------------------------------
result_key = f"result_{name}_{idx}"

if run:
    invalid_notice = (
        "notice_returned" in values
        and values["notice_returned"] > values["notice_sent"]
    )

    if invalid_notice:
        st.session_state.pop(result_key, None)
        st.error(
            "A returned notice requires a sent notice. "
            "Adjust the inputs and run the analysis again."
        )
    else:
        st.session_state[result_key] = values.copy()

# --------------------------------------------------
# RESULTS AND SUMMARY
# --------------------------------------------------
if result_key in st.session_state:
    scenario = pd.DataFrame(
        [st.session_state[result_key]],
        columns=spec["features"],
    )

    updated_probability = float(
        bundle["pipe"].predict_proba(scenario)[0, 1]
    )
    difference = (
        updated_probability - baseline_probability
    ) * 100

    st.divider()
    st.subheader("Scenario Results")

    left, middle, right = st.columns(3)

    with left:
        st.metric("Baseline Risk", f"{baseline_probability:.2%}")

    with middle:
        st.metric("Updated Risk", f"{updated_probability:.2%}")

    with right:
        st.metric(
            "Change From Baseline",
            f"{difference:+.2f} percentage points",
        )

    st.caption(
        "Differences use unrounded predictions. Results reflect "
        "the last submitted scenario. After editing inputs, "
        "run the analysis again to update them."
    )

    st.dataframe(
        scenario.rename(columns=label),
        hide_index=True,
        width="stretch",
    )

    st.subheader("Automatically Generated Scenario Summary")

    summary = summarize(
        name,
        baseline,
        scenario,
        baseline_probability,
        updated_probability,
    )

    for column in spec["features"]:
        summary = summary.replace(column, label(column))

    st.write(summary)

    st.download_button(
        "Download Scenario Summary",
        data=summary,
        file_name="scenario_summary.txt",
        mime="text/plain",
        key=f"download_{name}_{idx}",
    )

    # --------------------------------------------------
    # INDIVIDUAL CONTRIBUTIONS
    # --------------------------------------------------
    st.subheader("Contributions to This Prediction")

    reference, contributions = explain(bundle, scenario)

    st.caption(
        "These are exact additive logistic-model contributions "
        "relative to mean training features. This training reference "
        "is different from your selected baseline episode."
    )

    total_contribution = float(
        contributions["log_odds_contribution"].sum()
    )

    st.write(
        f"Reference log odds: **{reference:.4f}**. "
        f"Sum of contributions: **{total_contribution:+.4f}**. "
        f"Scenario log odds: "
        f"**{reference + total_contribution:.4f}**."
    )

    st.caption(
        "Adding the contributions to the reference reconstructs "
        "the scenario log odds. The logistic transformation converts "
        "that total into the displayed predicted probability."
    )

    display_contributions = contributions.copy()
    display_contributions["predictor"] = (
        display_contributions["predictor"].map(label)
    )
    display_contributions["Odds Multiplier"] = np.exp(
        display_contributions["log_odds_contribution"]
    )
    display_contributions = (
        display_contributions.reset_index(drop=True)
    )

    st.dataframe(
        display_contributions.rename(
            columns={
                "predictor": "Predictor",
                "log_odds_contribution": "Contribution in Log Odds",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    contribution_chart = (
        alt.Chart(display_contributions)
        .mark_bar()
        .encode(
            x=alt.X(
                "log_odds_contribution:Q",
                title="Contribution in Log Odds",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "predictor:N",
                title=None,
                sort=display_contributions["predictor"].tolist(),
                axis=alt.Axis(labelLimit=300),
            ),
            color=alt.condition(
                alt.datum.log_odds_contribution >= 0,
                alt.value("#D97706"),
                alt.value("#178A91"),
            ),
            tooltip=[
                alt.Tooltip("predictor:N", title="Predictor"),
                alt.Tooltip(
                    "log_odds_contribution:Q",
                    title="Log-Odds Contribution",
                    format=".4f",
                ),
                alt.Tooltip(
                    "Odds Multiplier:Q",
                    title="Odds Multiplier",
                    format=".4f",
                ),
            ],
        )
        .properties(
            height=max(150, len(display_contributions) * 40)
        )
    )

    st.altair_chart(contribution_chart, width="stretch")

    st.caption(
        "Orange bars indicate positive contributions; teal bars "
        "indicate negative contributions. Hover over a bar "
        "to compare its exact value with the table."
    )

    st.markdown("#### Interpreting This Prediction")

    for _, row in contributions.iterrows():
        contribution = float(row["log_odds_contribution"])
        multiplier = float(np.exp(contribution))

        if np.isclose(contribution, 0):
            direction = "makes approximately no contribution"
        elif contribution > 0:
            direction = "contributes toward higher predicted risk"
        else:
            direction = "contributes toward lower predicted risk"

        st.write(
            f"**{label(row['predictor'])}:** "
            f"{contribution:+.4f} log odds, corresponding to "
            f"a {multiplier:.4f}× odds multiplier for this component "
            f"relative to its training reference. It {direction}."
        )

    st.info(
        "Local contributions are measured in log odds, not "
        "percentage points. They explain this scenario relative "
        "to the training reference, not the change from the "
        "selected episode baseline. They describe model "
        "associations and do not establish intervention effects."
    )

    st.warning(
        "Use these results for illustrative, supportive planning. "
        "Input combinations may be uncommon, and predictions may "
        "not generalize to real populations. A changed input is "
        "not a guarantee of a changed real-world outcome."
    )