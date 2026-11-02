from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from decision_support import (
    FIXED_HISTORY,
    INPUT_ROLES,
    calculate_complexity,
    explain,
    input_comparison,
    label,
    prepare_scenario,
    scenario_columns,
    summarize_comparison,
    train_selected,
)


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "docs" / "evaluation"


@st.cache_resource
def fitted_models():
    return train_selected()


def show_contributions(bundle, row):
    """Explain one prediction relative to mean training features."""
    reference, contributions = explain(
        bundle, row[bundle["features"]]
    )

    table = contributions.copy()
    table["Predictor"] = table["predictor"].map(label)
    table["Odds Multiplier"] = np.exp(
        table["log_odds_contribution"]
    )

    st.caption(
        "This explains one prediction relative to mean training "
        "features. It is different from overall predictor importance "
        "and from the selected episode baseline."
    )

    st.dataframe(
        table[
            [
                "Predictor",
                "log_odds_contribution",
                "Odds Multiplier",
            ]
        ].rename(
            columns={
                "log_odds_contribution": "Contribution in Log Odds"
            }
        ),
        hide_index=True,
        width="stretch",
    )

    chart = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X(
                "log_odds_contribution:Q",
                title="Contribution in Log Odds",
                scale=alt.Scale(zero=True),
            ),
            y=alt.Y(
                "Predictor:N",
                sort=table["Predictor"].tolist(),
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            color=alt.condition(
                alt.datum.log_odds_contribution > 0,
                alt.value("#D97706"),
                alt.value("#178A91"),
            ),
            tooltip=[
                "Predictor:N",
                alt.Tooltip(
                    "log_odds_contribution:Q",
                    title="Contribution",
                    format=".4f",
                ),
                alt.Tooltip(
                    "Odds Multiplier:Q",
                    format=".4f",
                ),
            ],
        )
        .properties(height=max(130, 38 * len(table)))
    )

    st.altair_chart(chart, width="stretch")

    for _, item in table.iterrows():
        value = float(item["log_odds_contribution"])

        if np.isclose(value, 0):
            direction = "makes approximately no contribution"
        elif value > 0:
            direction = "moves the model toward higher risk"
        else:
            direction = "moves the model toward lower risk"

        st.write(
            f"**{item['Predictor']}:** {value:+.4f} log odds; "
            f"{direction} relative to its training reference."
        )

    with st.expander("How the explanation adds up"):
        total = float(contributions["log_odds_contribution"].sum())
        st.write(
            f"Reference log odds: {reference:.4f}. "
            f"Total contribution: {total:+.4f}. "
            f"Prediction log odds: {reference + total:.4f}."
        )
        st.write(
            "The logistic transformation converts the final log odds "
            "into probability. These contributions are not "
            "percentage-point effects and do not directly attribute "
            "the baseline-to-scenario difference."
        )

    st.info(
        "Positive means higher model risk relative to the training "
        "reference; negative means lower model risk. Contributions "
        "describe model associations, not causes or intervention effects."
    )


st.title("Renewal Risk and Outreach Scenarios")
st.warning(
    "Synthetic educational data. Use predictions for illustrative "
    "supportive planning, not eligibility or denial decisions."
)

try:
    with st.spinner("Preparing the evaluated models..."):
        data, bundles = fitted_models()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Unable to prepare models: {error}")
    st.stop()

name = st.selectbox(
    "Select a prediction model",
    list(bundles),
)
bundle = bundles[name]
columns = scenario_columns(name)

if name == "Coverage churn":
    st.markdown(
        "**KPI: Coverage churn rate** — the proportion of renewal "
        "episodes with a simulated coverage interruption within "
        "30 days after renewal."
    )
    st.caption(
        "Prediction time: 30 days before renewal. "
        "Intended use: coverage-continuity review and workload planning."
    )
else:
    st.markdown(
        "**KPI: Early outreach nonresponse rate** — the proportion "
        "without a response by 30 days before renewal, including "
        "episodes with no outreach attempts."
    )
    st.caption(
        "Forecast origin: 45 days before renewal. "
        "Contact Stability is assumed available then. "
        "The recorded dataset prediction date is day −30."
    )

st.caption(
    "Individual probabilities may be averaged over an appropriate "
    "cohort to estimate its expected outcome rate. "
    "No financial savings or causal intervention effects are estimated."
)

# --------------------------------------------------
# PERFORMANCE AND LIMITATIONS
# --------------------------------------------------
with st.expander("Model performance and limitations", expanded=True):
    st.write(
        "We compared the original and revised models on separate "
        "validation households. We selected models using validation "
        "log loss, then evaluated them on the original test households."
    )

    comparison = bundle["comparison"].copy()

    st.dataframe(
        comparison[
            [
                "version",
                "retained",
                "log_loss",
                "Brier",
                "ROC_AUC",
                "average_precision",
                "precision_top20",
                "recall_top20",
            ]
        ].rename(
            columns={
                "version": "Model",
                "retained": "Selected",
                "log_loss": "Log Loss",
                "Brier": "Brier Score",
                "ROC_AUC": "ROC-AUC",
                "average_precision": "Average Precision",
                "precision_top20": "Outcome Rate in Top 20%",
                "recall_top20": "Share of Outcomes in Top 20%",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    retained = comparison[
        comparison["version"].eq(
            bundle["selection"]["retained_version"]
        )
    ].iloc[0]
    constant = comparison[
        comparison["version"].eq("Constant baseline")
    ].iloc[0]

    improvement = (
        (constant["log_loss"] - retained["log_loss"])
        / constant["log_loss"]
        * 100
    )

    st.write(
        f"Compared with giving everyone the same training-rate "
        f"prediction, the selected model reduces test log loss by "
        f"about **{improvement:.1f}%**. Lower log loss and Brier score "
        "mean less probability error."
    )

    st.write(
        f"Under an illustrative capacity to review 20% of cases, "
        f"**{retained['precision_top20']:.1%}** of reviewed cases "
        f"have the outcome, capturing "
        f"**{retained['recall_top20']:.1%}** of all outcomes."
    )

    st.caption(
        "The 20% capacity is illustrative, not a staffing recommendation. "
        "Equal-risk cases are ordered by episode ID, which has no "
        "substantive decision meaning. The constant baseline cannot "
        "meaningfully rank cases."
    )

    st.write(
        "ROC-AUC describes ranking ability. Average precision summarizes "
        "positive-outcome ranking and should be compared with outcome "
        "prevalence. Neither establishes that an individual prediction "
        "is correct."
    )

    if name == "Coverage churn":
        st.write(
            "**Selected model:** logistic regression with stronger "
            "regularization, C=0.1. The revision changes probability "
            "error only slightly and slightly reduces ranking metrics. "
            "It is not a major accuracy improvement."
        )
        st.write(
            "**Practical limitation:** a low churn probability does "
            "not rule out coverage interruption. Do not exclude a case "
            "from assistance solely because its predicted risk is low."
        )
    else:
        st.write(
            "**Selected model:** logistic regression using Contact "
            "Stability only. Removing Address Stability simplifies "
            "explanations and slightly improves probability error, "
            "but test average precision and top-20% yield decrease."
        )
        st.write(
            "**Practical limitation:** cases in the same Contact "
            "Stability category receive the same probability. "
            "The model cannot reliably prioritize people within "
            "that category."
        )

    st.warning(
        "Performance is modest. The original test results were already "
        "inspected in the previous submission; this is a reused "
        "synthetic evaluation set, not untouched, temporal, or external "
        "validation. Model choices were fixed before this evaluation."
    )

    with st.expander("Calibration and error diagnostics"):
        slug = bundle["target"]
        version = bundle["selection"]["retained_version"].lower()

        calibration_path = (
            EVALUATION / f"{slug}_final_{version}_calibration.csv"
        )
        segment_path = (
            EVALUATION / f"{slug}_final_{version}_segments.csv"
        )

        if calibration_path.is_file():
            st.markdown("**Predicted versus observed probabilities**")
            st.dataframe(
                pd.read_csv(calibration_path),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Compare mean predictions with observed outcome rates "
                "within each probability interval. Differences are "
                "sample estimates, not proof of future calibration."
            )

        if segment_path.is_file():
            st.markdown("**Errors across case segments**")
            st.dataframe(
                pd.read_csv(segment_path),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Higher segment Brier scores can reflect different "
                "outcome rates as well as prediction error. "
                "They do not independently prove miscalibration "
                "or unfairness. These are retrospective diagnostics. "
                "For nonresponse, later administrative fields shown "
                "here are not forecasting inputs."
            )

# --------------------------------------------------
# OVERALL IMPORTANCE
# --------------------------------------------------
with st.expander("Overall Predictor Importance"):
    importance = bundle["importance"].copy()
    importance["Predictor"] = importance["predictor"].map(label)

    st.caption(
        "Predictive usefulness across the reused test cases: mean "
        "increase in log loss after shuffling an input, using five "
        "repetitions. Larger positive values indicate greater usefulness. "
        "Near-zero or negative values show little evidence of usefulness."
    )

    st.dataframe(
        importance[
            ["Predictor", "importance", "sd"]
        ].rename(
            columns={
                "importance": "Mean Increase in Log Loss",
                "sd": "Standard Deviation",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    chart = (
        alt.Chart(importance)
        .mark_bar(color="#178A91")
        .encode(
            x=alt.X(
                "importance:Q",
                title="Mean Increase in Log Loss",
            ),
            y=alt.Y(
                "Predictor:N",
                sort=importance["Predictor"].tolist(),
                title=None,
                axis=alt.Axis(labelLimit=300),
            ),
            tooltip=[
                "Predictor:N",
                alt.Tooltip("importance:Q", format=".6f"),
                alt.Tooltip("sd:Q", format=".6f"),
            ],
        )
        .properties(height=max(130, 38 * len(importance)))
    )

    st.altair_chart(chart, width="stretch")
    st.info(
        "Overall importance describes many cases. Individual "
        "contributions explain one prediction. Neither establishes "
        "causation, and correlated inputs may share importance."
    )

# --------------------------------------------------
# BASELINE
# --------------------------------------------------
st.divider()
st.subheader("Baseline Case")

idx = st.selectbox(
    "Select a held-out renewal episode",
    bundle["test"][:100].tolist(),
    format_func=lambda value: str(data.loc[value, "episode_id"]),
    key=f"baseline_{name}",
)

baseline = data.loc[[idx], columns].copy()
baseline_risk = float(
    bundle["pipe"].predict_proba(
        baseline[bundle["features"]]
    )[0, 1]
)

st.metric("Baseline Predicted Risk", f"{baseline_risk:.2%}")

st.dataframe(
    baseline.rename(columns=label),
    hide_index=True,
    width="stretch",
)

st.caption(
    "The baseline is the selected episode's original information. "
    "Its prediction stays fixed while scenario assumptions change. "
    "Probability is not certainty."
)

with st.expander("Explain the Baseline Prediction"):
    show_contributions(bundle, baseline)

# --------------------------------------------------
# INPUT ROLES
# --------------------------------------------------
with st.expander("Which inputs can reasonably change?", expanded=True):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Input": label(column),
                    "Role": INPUT_ROLES[column],
                }
                for column in columns
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    if name == "Coverage churn":
        st.write(
            "Historical coverage gaps and prior documentation issues "
            "are fixed for the selected episode. Notice status and "
            "outreach activity describe operational conditions, but "
            "their observed history cannot be changed retroactively."
        )
        st.write(
            "Beneficiary responses, returned mail, and verification "
            "requirements are not freely controlled actions. Edits "
            "represent alternative case assumptions at the same "
            "prediction snapshot, not a forecast of an intervention."
        )
    else:
        st.write(
            "Contact Stability describes a household condition. "
            "Changing it compares alternative profiles; it does not "
            "simulate the causal effect of updating a phone number. "
            "Address Stability was removed from the selected model."
        )

# --------------------------------------------------
# SCENARIO FORM
# --------------------------------------------------
st.subheader("What-if Scenario")

with st.form(f"scenario_{name}_{idx}"):
    st.caption(
        "Change assumptions and submit. Binary fields use "
        "0 = No and 1 = Yes. Unchanged scenarios are valid."
    )

    submitted = {}

    for column in columns:
        original = baseline.iloc[0][column]

        if column == "administrative_complexity":
            continue

        if column in FIXED_HISTORY:
            st.text_input(
                label(column),
                value=str(original),
                disabled=True,
                key=f"fixed_{name}_{idx}_{column}",
            )
            continue

        if pd.api.types.is_numeric_dtype(data[column]):
            submitted[column] = st.number_input(
                label(column),
                min_value=int(data[column].min()),
                max_value=int(data[column].max()),
                value=int(original),
                step=1,
                help=INPUT_ROLES[column],
                key=f"input_{name}_{idx}_{column}",
            )
        else:
            options = sorted(data[column].dropna().unique().tolist())
            submitted[column] = st.selectbox(
                label(column),
                options,
                index=options.index(original),
                help=INPUT_ROLES[column],
                key=f"input_{name}_{idx}_{column}",
            )

    if name == "Coverage churn":
        preview = baseline.iloc[0].to_dict()
        preview.update(submitted)
        st.caption(
            "Administrative Complexity is calculated automatically "
            "from prior gaps, documentation issues, and the income "
            "verification requirement. The submitted result will "
            "display its recalculated value."
        )

    run = st.form_submit_button("Run What-if Analysis")

result_key = f"validated_result_{name}_{idx}"

if run:
    scenario, errors, warnings = prepare_scenario(
        name, baseline, submitted, data
    )

    if errors:
        st.session_state.pop(result_key, None)
        for message in errors:
            st.error(message)
        st.info(
            "No scenario prediction was produced. Correct the inputs "
            "and submit again. Previous results for this case were cleared."
        )
    else:
        st.session_state[result_key] = {
            "values": scenario.iloc[0].to_dict(),
            "warnings": warnings,
        }

# --------------------------------------------------
# SIDE-BY-SIDE RESULTS
# --------------------------------------------------
if result_key in st.session_state:
    saved = st.session_state[result_key]
    scenario = pd.DataFrame([saved["values"]], columns=columns)

    updated_risk = float(
        bundle["pipe"].predict_proba(
            scenario[bundle["features"]]
        )[0, 1]
    )
    difference = (updated_risk - baseline_risk) * 100

    st.divider()
    st.subheader("Scenario Results")

    left, middle, right = st.columns(3)

    with left:
        st.metric("Baseline Risk", f"{baseline_risk:.2%}")
    with middle:
        st.metric("Updated Risk", f"{updated_risk:.2%}")
    with right:
        st.metric(
            "Change From Baseline",
            f"{difference:+.2f} percentage points",
        )

    st.subheader("Baseline and Scenario Inputs")

    st.dataframe(
        input_comparison(baseline, scenario),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "Results reflect the last valid submitted scenario. "
        "After editing, submit again to update them. Differences "
        "use unrounded predictions. Context fields may affect "
        "validation or derived inputs without directly entering the model."
    )

    for message in saved["warnings"]:
        st.warning(message)

    st.subheader("Automatically Generated Scenario Summary")

    summary = summarize_comparison(
        name,
        baseline,
        scenario,
        baseline_risk,
        updated_risk,
        saved["warnings"],
    )
    st.write(summary)

    st.download_button(
        "Download Scenario Summary",
        data=summary,
        file_name="scenario_summary.txt",
        mime="text/plain",
        key=f"download_{name}_{idx}",
    )

    st.subheader("Contributions to This Prediction")
    show_contributions(bundle, scenario)

    st.warning(
        "A lower scenario prediction is not proof that an action "
        "will improve an outcome. Consider staff workload, contact "
        "burden, omitted barriers, and uncertainty. These models "
        "do not estimate financial savings or establish eligibility."
    )