from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from decision_support import (
    calculate_complexity,
    explain,
    prepare_scenario,
    scenario_columns,
    train_selected,
)


ROOT = Path(__file__).resolve().parent
TIMEOUT = 180


def find(elements, label):
    matches = [item for item in elements if item.label == label]
    assert len(matches) == 1, (
        f"Expected one element labeled {label!r}; found {len(matches)}."
    )
    return matches[0]


def check_clean(app):
    assert not app.exception, str(app.exception)
    assert not app.error, "\n".join(item.value for item in app.error)


def metric(app, label):
    return find(app.metric, label).value


def submit(app):
    find(app.button, "Run What-if Analysis").click()
    app.run(timeout=TIMEOUT)
    assert not app.exception, str(app.exception)


def check_results(app):
    check_clean(app)

    for name in ["Baseline Risk", "Updated Risk", "Change From Baseline"]:
        metric(app, name)

    tables = [item.value for item in app.dataframe]

    assert any(
        {"Baseline", "Scenario", "Changed", "Input role"}.issubset(
            table.columns
        )
        for table in tables
    ), "Side-by-side input comparison is missing."

    assert any(
        "Contribution in Log Odds" in table.columns
        for table in tables
    ), "Individual explanation table is missing."

    text = "\n".join(item.value for item in app.markdown)

    for phrase in [
        "Scenario inputs:",
        "Changed inputs:",
        "Baseline predicted risk:",
        "Updated predicted risk:",
        "Change from baseline:",
        "Practical implications:",
        "Trade-offs:",
        "Limitations:",
    ]:
        assert phrase in text, f"Missing summary content: {phrase}"


def check_rejected(app, message):
    assert any(message in item.value for item in app.error)
    assert not any(
        item.label == "Updated Risk" for item in app.metric
    ), "Previous scenario prediction remains after invalid input."


# --------------------------------------------------
# MODEL AND SCENARIO CHECKS
# --------------------------------------------------
data, bundles = train_selected()

for name, bundle in bundles.items():
    index = bundle["test"][0]
    baseline = data.loc[[index], scenario_columns(name)].copy()

    scenario, errors, _ = prepare_scenario(
        name, baseline, {}, data
    )
    assert not errors, errors

    features = bundle["features"]
    p0 = bundle["pipe"].predict_proba(baseline[features])[0, 1]
    p1 = bundle["pipe"].predict_proba(scenario[features])[0, 1]

    assert np.isclose(p0, p1), "Unchanged inputs changed risk."

    reference, contributions = explain(bundle, scenario[features])
    reconstructed = reference + contributions[
        "log_odds_contribution"
    ].sum()
    actual = bundle["pipe"].decision_function(scenario[features])[0]

    assert np.isclose(reconstructed, actual)
    print(f"PASS: prediction consistency and explanation — {name}")

churn_base = data.loc[
    [bundles["Coverage churn"]["test"][0]],
    scenario_columns("Coverage churn"),
].copy()

changed_gap = int(churn_base.iloc[0]["prior_coverage_gaps"]) + 1
_, errors, _ = prepare_scenario(
    "Coverage churn",
    churn_base,
    {"prior_coverage_gaps": changed_gap},
    data,
)
assert errors, "Fixed history change was accepted."
print("PASS: fixed history is protected")

scenario, errors, _ = prepare_scenario(
    "Coverage churn",
    churn_base,
    {
        "income_verification_required": (
            1 - int(churn_base.iloc[0]["income_verification_required"])
        )
    },
    data,
)
assert not errors, errors
assert scenario.iloc[0]["administrative_complexity"] == (
    calculate_complexity(scenario.iloc[0])
)
print("PASS: complexity is recalculated")

_, errors, _ = prepare_scenario(
    "Coverage churn",
    churn_base,
    {"outreach_attempts_pre_prediction": 999},
    data,
)
assert errors, "Out-of-range input was accepted."
print("PASS: numeric range validation")

_, errors, _ = prepare_scenario(
    "Early outreach nonresponse",
    data.loc[
        [bundles["Early outreach nonresponse"]["test"][0]],
        scenario_columns("Early outreach nonresponse"),
    ],
    {"contact_stability": "Invalid category"},
    data,
)
assert errors, "Unknown category was accepted."
print("PASS: categorical validation")


# --------------------------------------------------
# PAGE LOADING
# --------------------------------------------------
for page in [
    "Renewal_Overview.py",
    "pages/2_Outreach_and_Case_Review.py",
    "pages/3_Predictive_Scenarios.py",
]:
    app = AppTest.from_file(
        str(ROOT / page), default_timeout=TIMEOUT
    ).run()
    check_clean(app)
    print(f"PASS: page loads — {page}")


# --------------------------------------------------
# PREDICTIVE WORKFLOWS
# --------------------------------------------------
app = AppTest.from_file(
    str(ROOT / "pages/3_Predictive_Scenarios.py"),
    default_timeout=TIMEOUT,
).run()
check_clean(app)

for name in ["Coverage churn", "Early outreach nonresponse"]:
    find(app.selectbox, "Select a prediction model").select(name)
    app.run(timeout=TIMEOUT)
    check_clean(app)

    submit(app)
    check_results(app)

    baseline_risk = metric(app, "Baseline Risk")
    assert baseline_risk == metric(app, "Updated Risk")
    assert metric(app, "Change From Baseline") in {
        "+0.00 percentage points",
        "-0.00 percentage points",
        "0.00 percentage points",
    }
    print(f"PASS: unchanged scenario — {name}")

    if name == "Coverage churn":
        assert find(app.text_input, "Prior Coverage Gaps").disabled
        assert find(app.text_input, "Prior Documentation Issues").disabled

        find(
            app.number_input, "Outreach Attempts Before Prediction"
        ).set_value(1)

        response = find(
            app.number_input, "Responded Before Prediction"
        )
        response.set_value(1 - int(response.value))
    else:
        stability = find(app.selectbox, "Contact Stability")
        alternative = next(
            option
            for option in stability.options
            if option != stability.value
        )
        stability.select(alternative)

    submit(app)
    check_results(app)

    assert metric(app, "Baseline Risk") == baseline_risk

    comparisons = [
        item.value
        for item in app.dataframe
        if "Changed" in item.value.columns
    ]
    assert comparisons[0]["Changed"].eq("Yes").any()

    # Modified inputs are allowed to leave predictions unchanged.
    print(f"PASS: modified scenario and fixed baseline — {name}")


# --------------------------------------------------
# INVALID INPUTS, STALE RESULTS, AND RECOVERY
# --------------------------------------------------
find(app.selectbox, "Select a prediction model").select(
    "Coverage churn"
)
app.run(timeout=TIMEOUT)
check_clean(app)

find(app.number_input, "Renewal Notice Sent").set_value(0)
find(app.number_input, "Renewal Notice Returned").set_value(1)
submit(app)
check_rejected(app, "A returned notice requires a sent notice.")
print("PASS: invalid notice rejected and old results cleared")

find(app.number_input, "Renewal Notice Sent").set_value(1)
find(
    app.number_input, "Outreach Attempts Before Prediction"
).set_value(1)
submit(app)
check_results(app)
print("PASS: recovery after correcting notice inputs")

find(
    app.number_input, "Outreach Attempts Before Prediction"
).set_value(0)
find(app.number_input, "Responded Before Prediction").set_value(1)
submit(app)
check_rejected(
    app,
    "A recorded response requires at least one outreach attempt.",
)
print("PASS: response without outreach rejected")

find(
    app.number_input, "Outreach Attempts Before Prediction"
).set_value(1)
submit(app)
check_results(app)
print("PASS: recovery after correcting outreach inputs")

print("\nAll updated application checks passed.")