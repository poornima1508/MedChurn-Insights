from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent

SPECS = {
    "Coverage churn": {
        "target": "churn_30d",
        "features": [
            "prior_coverage_gaps",
            "notice_sent",
            "notice_returned",
            "responded_pre_prediction",
            "prior_documentation_issues",
            "administrative_complexity",
        ],
        "time": (
            "30 days before renewal → interruption within "
            "30 days after renewal"
        ),
    },
    "Early outreach nonresponse": {
        "target": "nonresponse",
        "features": [
            "contact_stability",
            "address_stability",
        ],
        "time": (
            "45 days before renewal → no response by "
            "30 days before renewal"
        ),
    },
}


def load_data():
    """Load and validate the supplied synthetic modeling data."""
    data_path = (
        ROOT
        / "data"
        / "final"
        / "medicaid_modeling_dataset.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Modeling dataset not found: {data_path}"
        )

    data = pd.read_csv(data_path)

    required = {
        "household_id",
        "episode_id",
        "churn_30d",
        "responded_pre_prediction",
    }

    for spec in SPECS.values():
        required.update(spec["features"])

    missing = sorted(required.difference(data.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if data.empty:
        raise ValueError("The modeling dataset is empty.")

    if data[list(required)].isna().any().any():
        raise ValueError(
            "Required modeling columns contain missing values."
        )

    if not data["episode_id"].is_unique:
        raise ValueError("Episode IDs must be unique.")

    binary_columns = [
        "churn_30d",
        "notice_sent",
        "notice_returned",
        "responded_pre_prediction",
        "prior_documentation_issues",
    ]

    for column in binary_columns:
        if not data[column].isin([0, 1]).all():
            raise ValueError(
                f"{column} must contain only 0 or 1."
            )

    if (
        data["notice_returned"] > data["notice_sent"]
    ).any():
        raise ValueError(
            "A returned notice requires a sent notice."
        )

    data["nonresponse"] = (
        1 - data["responded_pre_prediction"]
    )

    return data


def train_all():
    """Fit and evaluate a separate model for each KPI."""
    data = load_data()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train, test = next(
        splitter.split(
            data,
            groups=data["household_id"],
        )
    )

    train_households = set(
        data.iloc[train]["household_id"]
    )
    test_households = set(
        data.iloc[test]["household_id"]
    )

    if train_households.intersection(test_households):
        raise ValueError(
            "Training and testing households overlap."
        )

    bundles = {}

    for name, spec in SPECS.items():
        columns = spec["features"]

        X = data[columns].copy()
        y = data[spec["target"]]

        # Supports pandas object, string and categorical dtypes.
        categorical_columns = [
            column
            for column in columns
            if not pd.api.types.is_numeric_dtype(X[column])
        ]

        numeric_columns = [
            column
            for column in columns
            if column not in categorical_columns
        ]

        if (
            y.iloc[train].nunique() != 2
            or y.iloc[test].nunique() != 2
        ):
            raise ValueError(
                f"{name}: both outcomes must occur in "
                "training and testing data."
            )

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "num",
                    StandardScaler(),
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop=None,
                        sparse_output=False,
                    ),
                    categorical_columns,
                )
            )

        preprocessing = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        pipeline = Pipeline(
            steps=[
                ("pre", preprocessing),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        C=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )

        pipeline.fit(
            X.iloc[train],
            y.iloc[train],
        )

        probabilities = pipeline.predict_proba(
            X.iloc[test]
        )[:, 1]

        constant_probabilities = np.full(
            len(test),
            float(y.iloc[train].mean()),
        )

        metrics = {
            "test_rows": int(len(test)),
            "train_rows": int(len(train)),
            "test_prevalence": float(
                y.iloc[test].mean()
            ),
            "ROC_AUC": float(
                roc_auc_score(
                    y.iloc[test],
                    probabilities,
                )
            ),
            "average_precision": float(
                average_precision_score(
                    y.iloc[test],
                    probabilities,
                )
            ),
            "Brier": float(
                brier_score_loss(
                    y.iloc[test],
                    probabilities,
                )
            ),
            "baseline_Brier": float(
                brier_score_loss(
                    y.iloc[test],
                    constant_probabilities,
                )
            ),
            "log_loss": float(
                log_loss(
                    y.iloc[test],
                    probabilities,
                )
            ),
        }

        permutation = permutation_importance(
            pipeline,
            X.iloc[test],
            y.iloc[test],
            scoring="neg_log_loss",
            n_repeats=5,
            random_state=42,
        )

        importance = pd.DataFrame(
            {
                "predictor": columns,
                "importance": permutation.importances_mean,
                "sd": permutation.importances_std,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        transformed_training = (
            pipeline.named_steps["pre"].transform(
                X.iloc[train]
            )
        )

        reference = np.asarray(
            transformed_training
        ).mean(axis=0)

        bundles[name] = {
            "pipe": pipeline,
            "metrics": metrics,
            "importance": importance,
            "reference": reference,
            "nums": numeric_columns,
            "cats": categorical_columns,
            "test": test,
            "train": train,
        }

    return data, bundles


def explain(bundle, row):
    """
    Explain one prediction using exact centered logistic
    contributions in log odds.

    Reference log odds + summed contributions equals
    the model's log odds for this row.
    """
    if len(row) != 1:
        raise ValueError(
            "Individual explanations require exactly one row."
        )

    preprocessing = bundle["pipe"].named_steps["pre"]
    model = bundle["pipe"].named_steps["model"]

    transformed = np.asarray(
        preprocessing.transform(row)
    )[0]

    coefficients = model.coef_[0]

    contributions = (
        transformed - bundle["reference"]
    ) * coefficients

    grouped = {}
    offset = 0

    for column in bundle["nums"]:
        grouped[column] = float(contributions[offset])
        offset += 1

    if bundle["cats"]:
        encoder = preprocessing.named_transformers_["cat"]

        for column, categories in zip(
            bundle["cats"],
            encoder.categories_,
        ):
            width = len(categories)

            grouped[column] = float(
                contributions[
                    offset:offset + width
                ].sum()
            )

            offset += width

    reference_log_odds = float(
        model.intercept_[0]
        + bundle["reference"] @ coefficients
    )

    result = pd.DataFrame(
        {
            "predictor": list(grouped.keys()),
            "log_odds_contribution": list(grouped.values()),
        }
    )

    reconstructed = (
        reference_log_odds
        + result["log_odds_contribution"].sum()
    )

    actual = float(
        bundle["pipe"].decision_function(row)[0]
    )

    if not np.isclose(reconstructed, actual):
        raise ValueError(
            "Explanation does not reconstruct the prediction."
        )

    return reference_log_odds, result


def summarize(name, baseline, scenario, p0, p1):
    """Generate a summary for every submitted valid scenario."""
    changes = []

    for column in baseline.columns:
        original = baseline.iloc[0][column]
        updated = scenario.iloc[0][column]

        if original != updated:
            changes.append(
                f"{column}: {original} → {updated}"
            )

    inputs = ", ".join(
        f"{column}={scenario.iloc[0][column]}"
        for column in scenario.columns
    )

    change_description = (
        "; ".join(changes)
        if changes
        else "none; all inputs match the baseline"
    )

    difference = (p1 - p0) * 100

    if np.isclose(p0, p1, rtol=0, atol=1e-12):
        comparison = (
            "The predicted risk is unchanged from baseline."
        )
    elif p1 > p0:
        comparison = (
            "The scenario has higher predicted risk "
            "than the baseline."
        )
    else:
        comparison = (
            "The scenario has lower predicted risk "
            "than the baseline."
        )

    if name == "Coverage churn":
        implication = (
            "Higher estimated churn risk can flag a case "
            "for supportive coverage-continuity review."
        )
        specific_limitation = (
            "The churn outcome is simulated rather than "
            "an observed enrollment event."
        )
    else:
        implication = (
            "Higher estimated nonresponse risk can support "
            "early outreach planning and workload review."
        )
        specific_limitation = (
            "Nonresponse includes episodes with no outreach "
            "attempts under the simulated historical outreach mix. "
            "This earlier forecast assumes household contact "
            "attributes are known 45 days before renewal."
        )

    return (
        f"Model: {name}.\n\n"
        f"Scenario inputs: {inputs}.\n\n"
        f"Changed inputs: {change_description}.\n\n"
        f"Baseline predicted risk: {p0:.2%}. "
        f"Updated predicted risk: {p1:.2%}. "
        f"Change from baseline: {difference:+.2f} "
        f"percentage points. {comparison}\n\n"
        f"Practical implications: {implication} "
        "Predictions do not establish eligibility or "
        "justify denial of services.\n\n"
        "Limitations: These are synthetic-data model scenarios, "
        "not guaranteed effects of real-world interventions. "
        "Changing an input does not establish causation. "
        "Some input combinations may be uncommon. "
        "The model may omit important barriers and may be "
        "miscalibrated on real populations. External validation "
        f"is required. {specific_limitation}"
    )