from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parent
TIMEOUT = 180


def check_no_exceptions(app):
    messages = "\n".join(
        str(exception.message)
        for exception in app.exception
    )
    assert not app.exception, messages


def check_no_errors(app, context="Application"):
    messages = "\n".join(
        str(error.value)
        for error in app.error
    )
    assert not app.error, f"{context} failed:\n{messages}"


def find_element(elements, label):
    matches = [
        element
        for element in elements
        if element.label == label
    ]

    assert len(matches) == 1, (
        f"Expected one element labeled {label!r}; "
        f"found {len(matches)}."
    )

    return matches[0]


def metric_value(app, label):
    return find_element(app.metric, label).value


def run_scenario(app):
    find_element(
        app.button,
        "Run What-if Analysis",
    ).click()

    app.run(timeout=TIMEOUT)
    check_no_exceptions(app)


def check_results(app):
    check_no_errors(app, "Scenario")

    for label in [
        "Baseline Risk",
        "Updated Risk",
        "Change From Baseline",
    ]:
        metric_value(app, label)

    headings = {
        heading.value
        for heading in app.subheader
    }

    assert "Scenario Results" in headings
    assert "Automatically Generated Scenario Summary" in headings
    assert "Contributions to This Prediction" in headings

    summary_text = "\n".join(
        block.value
        for block in app.markdown
    )

    for required in [
        "Scenario inputs:",
        "Baseline predicted risk:",
        "Updated predicted risk:",
        "Change from baseline:",
        "Practical implications:",
        "Limitations:",
    ]:
        assert required in summary_text, (
            f"Missing summary content: {required}"
        )

    assert any(
        "log_odds_contribution" in table.value.columns
        or "Contribution in Log Odds" in table.value.columns
        for table in app.dataframe
    ), "Individual contribution table is missing."


# --------------------------------------------------
# PAGE LOADING
# --------------------------------------------------
pages = [
    "Renewal_Overview.py",
    "pages/2_Outreach_and_Case_Review.py",
    "pages/3_Predictive_Scenarios.py",
]

for page in pages:
    page_path = ROOT / page

    assert page_path.exists(), (
        f"Application file not found: {page_path}"
    )

    app = AppTest.from_file(
        str(page_path),
        default_timeout=TIMEOUT,
    ).run()

    check_no_exceptions(app)
    check_no_errors(app, f"Page {page}")

    print(f"PASS: page loads — {page}")


# --------------------------------------------------
# PREDICTIVE WORKFLOWS
# --------------------------------------------------
app = AppTest.from_file(
    str(ROOT / "pages/3_Predictive_Scenarios.py"),
    default_timeout=TIMEOUT,
).run()

check_no_exceptions(app)
check_no_errors(app, "Predictive page")

for name in [
    "Coverage churn",
    "Early outreach nonresponse",
]:
    find_element(
        app.selectbox,
        "Select a prediction model",
    ).select(name)

    app.run(timeout=TIMEOUT)

    check_no_exceptions(app)
    check_no_errors(app, name)

    # Test original baseline inputs.
    run_scenario(app)
    check_results(app)

    baseline = metric_value(app, "Baseline Risk")
    updated = metric_value(app, "Updated Risk")
    difference = metric_value(app, "Change From Baseline")

    assert baseline == updated, (
        f"{name}: unchanged inputs changed the prediction."
    )

    assert difference in {
        "+0.00 percentage points",
        "0.00 percentage points",
        "-0.00 percentage points",
    }, (
        f"{name}: expected zero change, received {difference}."
    )

    print(f"PASS: unchanged scenario — {name}")

    # Modify one predictor.
    if name == "Coverage churn":
        response = find_element(
            app.number_input,
            "Responded Before Prediction",
        )
        response.set_value(1 - int(response.value))

    else:
        stability = find_element(
            app.selectbox,
            "Contact Stability",
        )

        alternative = next(
            option
            for option in stability.options
            if option != stability.value
        )

        stability.select(alternative)

    run_scenario(app)
    check_results(app)

    assert metric_value(app, "Baseline Risk") == baseline, (
        f"{name}: the selected baseline changed."
    )

    summary_text = "\n".join(
        block.value
        for block in app.markdown
    )

    assert "Changed inputs: none" not in summary_text, (
        f"{name}: summary did not identify changed inputs."
    )

    # Modified inputs are allowed to produce the same prediction.
    print(f"PASS: modified scenario — {name}")


# --------------------------------------------------
# INVALID NOTICE INPUTS
# --------------------------------------------------
find_element(
    app.selectbox,
    "Select a prediction model",
).select("Coverage churn")

app.run(timeout=TIMEOUT)

check_no_exceptions(app)
check_no_errors(app, "Coverage churn page")

find_element(
    app.number_input,
    "Renewal Notice Sent",
).set_value(0)

find_element(
    app.number_input,
    "Renewal Notice Returned",
).set_value(1)

run_scenario(app)

assert any(
    "A returned notice requires a sent notice"
    in error.value
    for error in app.error
), "Invalid notice combination was not rejected."

assert not any(
    metric.label == "Updated Risk"
    for metric in app.metric
), "Stale scenario results remain after invalid input."

print("PASS: invalid notice combination rejected")


# --------------------------------------------------
# RECOVERY AFTER CORRECTION
# --------------------------------------------------
find_element(
    app.number_input,
    "Renewal Notice Sent",
).set_value(1)

run_scenario(app)
check_results(app)

print("PASS: valid scenario works after correcting inputs")
print("All application checks passed.")