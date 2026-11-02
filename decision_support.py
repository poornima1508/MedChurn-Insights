"""Selected models and validated scenario comparisons."""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance

from modeling import load_data, explain
from evaluate_models import make_pipeline


ROOT = Path(__file__).resolve().parent
EVALUATION = ROOT / "docs" / "evaluation"

LABELS = {
    "prior_coverage_gaps": "Prior Coverage Gaps",
    "notice_sent": "Renewal Notice Sent",
    "notice_returned": "Renewal Notice Returned",
    "responded_pre_prediction": "Responded Before Prediction",
    "prior_documentation_issues": "Prior Documentation Issues",
    "income_verification_required": "Income Verification Required",
    "administrative_complexity": "Administrative Complexity",
    "outreach_attempts_pre_prediction": "Outreach Attempts Before Prediction",
    "contact_stability": "Contact Stability",
    "address_stability": "Address Stability",
}

CHURN_INPUTS = [
    "prior_coverage_gaps",
    "prior_documentation_issues",
    "income_verification_required",
    "administrative_complexity",
    "notice_sent",
    "notice_returned",
    "outreach_attempts_pre_prediction",
    "responded_pre_prediction",
]

NONRESPONSE_INPUTS = ["contact_stability"]

FIXED_HISTORY = {
    "prior_coverage_gaps",
    "prior_documentation_issues",
}

INPUT_ROLES = {
    "prior_coverage_gaps": "Fixed history",
    "prior_documentation_issues": "Fixed history",
    "income_verification_required": "External administrative condition",
    "administrative_complexity": "Calculated from underlying inputs",
    "notice_sent": "Operational status; scenario assumption",
    "notice_returned": "Observed delivery condition; scenario assumption",
    "outreach_attempts_pre_prediction": "Operational activity; scenario assumption",
    "responded_pre_prediction": "Beneficiary response; not directly controlled",
    "contact_stability": "Household condition; alternative-profile assumption",
}


def label(column):
    return LABELS.get(column, column.replace("_", " ").title())


def calculate_complexity(row):
    score = (
        int(row["prior_documentation_issues"])
        + int(row["income_verification_required"])
        + int(row["prior_coverage_gaps"] > 0)
    )
    return "Low" if score == 0 else "Medium" if score == 1 else "High"


def scenario_columns(name):
    return (
        CHURN_INPUTS
        if name == "Coverage churn"
        else NONRESPONSE_INPUTS
    )


def train_selected():
    """Fit the frozen selected specifications on original training rows."""
    data = load_data()

    selection_path = EVALUATION / "model_selection.json"
    record_path = EVALUATION / "final_evaluation_record.json"
    manifest_path = EVALUATION / "split_manifest.csv"
    metrics_path = EVALUATION / "final_test_comparison.csv"

    for path in [
        selection_path,
        record_path,
        manifest_path,
        metrics_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Evaluation output missing: {path.name}"
            )

    selections = json.loads(selection_path.read_text())
    record = json.loads(record_path.read_text())

    selection_hash = hashlib.sha256(
        selection_path.read_bytes()
    ).hexdigest()

    if selection_hash != record["selection_file_sha256"]:
        raise ValueError(
            "Model selections changed after final evaluation."
        )

    data_path = (
        ROOT / "data" / "final" / "medicaid_modeling_dataset.csv"
    )
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()

    if data_hash != record["dataset_sha256"]:
        raise ValueError(
            "Dataset differs from the evaluated dataset."
        )

    required = set(CHURN_INPUTS + NONRESPONSE_INPUTS)
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Missing scenario fields: {sorted(missing)}")

    if data[list(required)].isna().any().any():
        raise ValueError("Scenario fields contain missing values.")

    expected_complexity = data.apply(calculate_complexity, axis=1)

    if not expected_complexity.eq(
        data["administrative_complexity"]
    ).all():
        raise ValueError(
            "Administrative Complexity does not match its source fields."
        )

    if (
        data["responded_pre_prediction"].gt(0)
        & data["outreach_attempts_pre_prediction"].eq(0)
    ).any():
        raise ValueError("Response recorded without an outreach attempt.")

    manifest = pd.read_csv(manifest_path)

    if (
        not manifest["episode_id"].is_unique
        or set(manifest["episode_id"]) != set(data["episode_id"])
    ):
        raise ValueError("Split manifest does not match the dataset.")

    partitions = data["episode_id"].map(
        manifest.set_index("episode_id")["partition"]
    )

    train = np.flatnonzero(
        partitions.isin(["development_train", "validation"])
        .to_numpy()
    )
    test = np.flatnonzero(
        partitions.eq("original_test_reserved").to_numpy()
    )

    if len(train) != 22568 or len(test) != 7432:
        raise ValueError("Unexpected training or evaluation row counts.")

    if (
        set(data.iloc[train]["household_id"])
        & set(data.iloc[test]["household_id"])
    ):
        raise ValueError("Training and test households overlap.")

    comparison = pd.read_csv(metrics_path)
    bundles = {}

    for name, selection in selections.items():
        features = selection["features"]
        target = selection["target"]

        pipeline = make_pipeline(
            data, features, selection["C"]
        )
        pipeline.fit(
            data.iloc[train][features],
            data.iloc[train][target],
        )

        preprocessing = pipeline.named_steps["pre"]
        reference = np.asarray(
            preprocessing.transform(data.iloc[train][features])
        ).mean(axis=0)

        numeric = [
            column
            for column in features
            if pd.api.types.is_numeric_dtype(data[column])
        ]
        categorical = [
            column for column in features if column not in numeric
        ]

        # Descriptive explanation only; no further model selection.
        permutation = permutation_importance(
            pipeline,
            data.iloc[test][features],
            data.iloc[test][target],
            scoring="neg_log_loss",
            n_repeats=5,
            random_state=42,
        )

        importance = pd.DataFrame(
            {
                "predictor": features,
                "importance": permutation.importances_mean,
                "sd": permutation.importances_std,
            }
        ).sort_values("importance", ascending=False)

        bundles[name] = {
            "pipe": pipeline,
            "features": features,
            "target": target,
            "selection": selection,
            "comparison": comparison[
                comparison["KPI"].eq(name)
            ].copy(),
            "reference": reference,
            "nums": numeric,
            "cats": categorical,
            "importance": importance,
            "train": train,
            "test": test,
        }

    return data, bundles


def prepare_scenario(name, baseline, submitted, data):
    """Validate scenario inputs and recalculate derived fields."""
    columns = scenario_columns(name)
    original = baseline.iloc[0]
    values = original[columns].to_dict()
    values.update(submitted)
    errors = []
    warnings = []

    for column in columns:
        if column == "administrative_complexity":
            continue

        value = values[column]

        if pd.api.types.is_numeric_dtype(data[column]):
            try:
                number = float(value)
            except (ValueError, TypeError):
                errors.append(f"{label(column)} must be numeric.")
                continue

            if not np.isfinite(number) or not number.is_integer():
                errors.append(
                    f"{label(column)} must be a finite whole number."
                )
                continue

            if not data[column].min() <= number <= data[column].max():
                errors.append(
                    f"{label(column)} must be within the observed range."
                )
                continue

            values[column] = int(number)
        elif value not in data[column].dropna().unique():
            errors.append(f"Choose a valid {label(column)} category.")

    if errors:
        return None, errors, warnings

    if name == "Coverage churn":
        for column in FIXED_HISTORY:
            if values[column] != original[column]:
                errors.append(
                    f"{label(column)} is fixed history and cannot "
                    "be changed for this episode."
                )

        if values["notice_returned"] > values["notice_sent"]:
            errors.append(
                "A returned notice requires a sent notice."
            )

        if (
            values["responded_pre_prediction"] == 1
            and values["outreach_attempts_pre_prediction"] == 0
        ):
            errors.append(
                "A recorded response requires at least one outreach attempt."
            )

        values["administrative_complexity"] = calculate_complexity(
            values
        )

        if (
            values["income_verification_required"]
            != original["income_verification_required"]
        ):
            warnings.append(
                "Verification requirements are external conditions. "
                "This is an alternative case assumption, not an "
                "action the supervisor can freely choose."
            )

        if (
            values["responded_pre_prediction"]
            != original["responded_pre_prediction"]
        ):
            warnings.append(
                "A beneficiary response is not directly controlled "
                "by the supervisor."
            )

        if (
            values["notice_returned"] != original["notice_returned"]
        ):
            warnings.append(
                "Returned mail is a delivery condition, not a "
                "guaranteed result of an action."
            )
    elif values["contact_stability"] != original["contact_stability"]:
        warnings.append(
            "Contact Stability describes a household profile. "
            "Changing this category does not estimate the effect "
            "of an outreach intervention."
        )

    if errors:
        return None, errors, warnings

    scenario = pd.DataFrame(
        [{column: values[column] for column in columns}]
    )

    matches = pd.Series(True, index=data.index)
    for column in columns:
        matches &= data[column].eq(values[column])

    if matches.sum() < 10:
        warnings.append(
            "Fewer than 10 supplied episodes match this complete "
            "input profile. Treat the combination cautiously."
        )

    return scenario, errors, warnings


def input_comparison(baseline, scenario):
    rows = []

    for column in scenario.columns:
        before = baseline.iloc[0][column]
        after = scenario.iloc[0][column]

        rows.append(
            {
                "Predictor / context": label(column),
                "Baseline": str(before),
                "Scenario": str(after),
                "Changed": "Yes" if before != after else "No",
                "Input role": INPUT_ROLES.get(column, ""),
            }
        )

    return pd.DataFrame(rows)


def summarize_comparison(name, baseline, scenario, p0, p1, warnings):
    inputs = "; ".join(
        f"{label(column)}={scenario.iloc[0][column]}"
        for column in scenario.columns
    )

    changes = [
        f"{label(column)}: {baseline.iloc[0][column]} "
        f"→ {scenario.iloc[0][column]}"
        for column in scenario.columns
        if baseline.iloc[0][column] != scenario.iloc[0][column]
    ]

    difference = (p1 - p0) * 100

    if np.isclose(p0, p1, rtol=0, atol=1e-12):
        direction = "The predicted risk is unchanged."
    elif p1 > p0:
        direction = "The scenario has higher model-predicted risk."
    else:
        direction = "The scenario has lower model-predicted risk."

    if name == "Coverage churn":
        implication = (
            "The estimate can support coverage-continuity case review."
        )
        tradeoff = (
            "Additional outreach may require staff time and create "
            "contact burden. The model does not estimate its cost "
            "or whether it prevents churn."
        )
        specific = (
            "Churn labels are simulated, and some low-risk cases "
            "still have a churn outcome."
        )
    else:
        implication = (
            "The estimate can support early outreach workload planning."
        )
        tradeoff = (
            "Prioritizing one contact category may leave other cases "
            "with less attention. Cases in the same category receive "
            "the same model probability; the model cannot reliably "
            "rank them within that category."
        )
        specific = (
            "Nonresponse includes cases with no outreach attempts. "
            "Contact Stability is assumed known 45 days before renewal."
        )

    caution = " ".join(warnings) if warnings else (
        "No additional input warnings were triggered. "
        "Passing checks does not establish real-world plausibility."
    )

    return (
        f"Model: {name}\n\n"
        f"Scenario inputs: {inputs}.\n\n"
        f"Changed inputs: {'; '.join(changes) if changes else 'none'}.\n\n"
        f"Baseline predicted risk: {p0:.2%}.\n"
        f"Updated predicted risk: {p1:.2%}.\n"
        f"Change from baseline: {difference:+.2f} percentage points.\n"
        f"{direction}\n\n"
        f"Practical implications: {implication}\n\n"
        f"Trade-offs: {tradeoff} No financial savings are estimated.\n\n"
        f"Input cautions: {caution}\n\n"
        f"Limitations: {specific} These are synthetic-data scenarios, "
        "not proven causal effects or guaranteed outcomes. "
        "Evaluation reused the original test households and does not "
        "establish performance on future or real populations. "
        "Predictions must not determine eligibility or justify denial."
    )