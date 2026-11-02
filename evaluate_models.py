"""Compare planned model improvements using validation data only.

This script deliberately does not evaluate the original test set.
Final evaluation will follow after model choices are documented.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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

from modeling import SPECS, load_data


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "evaluation"
DATA_PATH = (
    ROOT / "data" / "final" / "medicaid_modeling_dataset.csv"
)


def make_pipeline(data, features, regularization):
    """Build preprocessing without fitting it on validation data."""
    numeric = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(data[column])
    ]
    categorical = [
        column for column in features if column not in numeric
    ]

    transformers = []

    if numeric:
        transformers.append(
            ("num", StandardScaler(), numeric)
        )

    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            )
        )

    return Pipeline(
        [
            ("pre", ColumnTransformer(transformers)),
            (
                "model",
                LogisticRegression(
                    C=regularization,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_metrics(y, probabilities, episode_ids):
    """Measure probability quality and illustrative review yield."""
    y = np.asarray(y, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    episode_ids = np.asarray(episode_ids, dtype=str)

    # Select exactly 20% of cases. Resolve probability ties using
    # episode ID so selection is deterministic, not outcome-based.
    review_count = max(1, int(np.ceil(len(y) * 0.20)))
    order = np.lexsort((episode_ids, -probabilities))
    selected = order[:review_count]
    positives = int(y.sum())

    return {
        "rows": int(len(y)),
        "prevalence": float(y.mean()),
        "ROC_AUC": (
            float(roc_auc_score(y, probabilities))
            if len(np.unique(y)) == 2
            else None
        ),
        "average_precision": (
            float(average_precision_score(y, probabilities))
            if positives
            else None
        ),
        "Brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(
            log_loss(y, probabilities, labels=[0, 1])
        ),
        "reviewed_cases": review_count,
        "precision_top20": float(y[selected].mean()),
        "recall_top20": (
            float(y[selected].sum() / positives)
            if positives
            else None
        ),
    }


def save_calibration(y, probabilities, filename):
    """Use fixed probability intervals for calibration diagnostics."""
    table = pd.DataFrame(
        {
            "observed": np.asarray(y),
            "predicted": probabilities,
        }
    )
    table["probability_band"] = pd.cut(
        table["predicted"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )

    result = (
        table.groupby("probability_band", observed=True)
        .agg(
            cases=("observed", "size"),
            observed_rate=("observed", "mean"),
            mean_prediction=("predicted", "mean"),
        )
        .reset_index()
    )

    result["prediction_minus_observed"] = (
        result["mean_prediction"] - result["observed_rate"]
    )
    result.to_csv(OUTPUT / filename, index=False)


def save_errors(data, rows, target, probabilities, prefix):
    """Save segment diagnostics and largest probability errors."""
    cases = data.iloc[rows].copy()
    cases["observed_outcome"] = cases[target].astype(int)
    cases["predicted_risk"] = probabilities
    cases["squared_error"] = (
        cases["predicted_risk"] - cases["observed_outcome"]
    ) ** 2
    cases["coverage_gap_segment"] = np.where(
        cases["prior_coverage_gaps"] > 0,
        "One or more",
        "Zero",
    )

    segments = []

    for column in [
        "contact_stability",
        "administrative_complexity",
        "coverage_gap_segment",
    ]:
        for value, part in cases.groupby(column, observed=True):
            segments.append(
                {
                    "segment_field": column,
                    "segment_value": value,
                    "cases": int(len(part)),
                    "observed_rate": float(
                        part["observed_outcome"].mean()
                    ),
                    "mean_prediction": float(
                        part["predicted_risk"].mean()
                    ),
                    "Brier": float(
                        part["squared_error"].mean()
                    ),
                    "small_segment": len(part) < 100,
                }
            )

    pd.DataFrame(segments).to_csv(
        OUTPUT / f"{prefix}_segments.csv",
        index=False,
    )

    columns = [
        "episode_id",
        "contact_stability",
        "administrative_complexity",
        "prior_coverage_gaps",
        "notice_sent",
        "notice_returned",
        "responded_pre_prediction",
        "observed_outcome",
        "predicted_risk",
        "squared_error",
    ]

    cases.nlargest(20, "squared_error")[columns].to_csv(
        OUTPUT / f"{prefix}_largest_errors.csv",
        index=False,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_data()

    # Exactly reproduce the original household partition.
    original_train, original_test = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.25,
            random_state=42,
        ).split(data, groups=data["household_id"])
    )

    development = data.iloc[original_train]

    inner_train, inner_validation = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=43,
        ).split(
            development,
            groups=development["household_id"],
        )
    )

    train = original_train[inner_train]
    validation = original_train[inner_validation]

    partitions = {
        "development_train": train,
        "validation": validation,
        "original_test_reserved": original_test,
    }

    household_sets = {
        name: set(data.iloc[rows]["household_id"])
        for name, rows in partitions.items()
    }

    names = list(household_sets)

    for position, first in enumerate(names):
        for second in names[position + 1:]:
            if household_sets[first] & household_sets[second]:
                raise ValueError(
                    f"Household overlap: {first} and {second}"
                )

    manifest = data[["episode_id", "household_id"]].copy()
    manifest["partition"] = ""

    for name, rows in partitions.items():
        manifest.loc[rows, "partition"] = name

    manifest.to_csv(
        OUTPUT / "split_manifest.csv",
        index=False,
    )

    metadata = {
        "dataset_sha256": hashlib.sha256(
            DATA_PATH.read_bytes()
        ).hexdigest(),
        "original_split_seed": 42,
        "validation_split_seed": 43,
        "partition_rows": {
            name: int(len(rows))
            for name, rows in partitions.items()
        },
        "household_overlap": False,
        "selection_metric": "validation log loss",
        "test_evaluated_by_this_script": False,
        "limitation": (
            "Original test results were previously inspected. "
            "The reserved partition is not a new untouched test."
        ),
    }

    with (OUTPUT / "evaluation_metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)

    results = []
    selections = {}

    for name, spec in SPECS.items():
        target = spec["target"]
        original_features = list(spec["features"])
        slug = target

        if name == "Coverage churn":
            revised_features = original_features
            revised_c = 0.1
            proposed_change = (
                "Stronger regularization: C=1 to C=0.1"
            )
        else:
            revised_features = ["contact_stability"]
            revised_c = 1.0
            proposed_change = (
                "Remove Address Stability; retain Contact Stability"
            )

        y_train = data.iloc[train][target]
        y_validation = data.iloc[validation][target]
        episode_ids = data.iloc[validation]["episode_id"]

        if y_train.nunique() != 2 or y_validation.nunique() != 2:
            raise ValueError(
                f"{name}: both outcomes required in each partition."
            )

        baseline_probabilities = np.full(
            len(validation),
            float(y_train.mean()),
        )

        baseline_metrics = calculate_metrics(
            y_validation,
            baseline_probabilities,
            episode_ids,
        )

        results.append(
            {
                "KPI": name,
                "version": "Constant baseline",
                **baseline_metrics,
            }
        )

        save_calibration(
            y_validation,
            baseline_probabilities,
            f"{slug}_baseline_calibration.csv",
        )

        candidates = {
            "Original": (original_features, 1.0),
            "Revised": (revised_features, revised_c),
        }

        candidate_metrics = {}

        for version, (features, regularization) in candidates.items():
            pipeline = make_pipeline(
                data,
                features,
                regularization,
            )

            pipeline.fit(
                data.iloc[train][features],
                y_train,
            )

            probabilities = pipeline.predict_proba(
                data.iloc[validation][features]
            )[:, 1]

            metrics = calculate_metrics(
                y_validation,
                probabilities,
                episode_ids,
            )

            candidate_metrics[version] = metrics

            results.append(
                {
                    "KPI": name,
                    "version": version,
                    **metrics,
                }
            )

            prefix = f"{slug}_{version.lower()}"

            save_calibration(
                y_validation,
                probabilities,
                f"{prefix}_calibration.csv",
            )

            save_errors(
                data,
                validation,
                target,
                probabilities,
                prefix,
            )

        original_loss = candidate_metrics["Original"]["log_loss"]
        revised_loss = candidate_metrics["Revised"]["log_loss"]

        retained = (
            "Revised"
            if revised_loss < original_loss
            else "Original"
        )

        retained_features, retained_c = candidates[retained]

        selections[name] = {
            "target": target,
            "proposed_change": proposed_change,
            "retained_version": retained,
            "features": retained_features,
            "C": retained_c,
            "original_validation_log_loss": original_loss,
            "revised_validation_log_loss": revised_loss,
            "revised_minus_original_log_loss": (
                revised_loss - original_loss
            ),
            "reason": (
                "Lower validation log loss."
                if revised_loss != original_loss
                else "Equal validation log loss; retain original."
            ),
            "practical_significance": (
                "Numerical selection only. Review the size of the "
                "difference and supporting metrics before claiming "
                "a meaningful operational improvement."
            ),
        }

    comparison = pd.DataFrame(results)
    comparison.to_csv(
        OUTPUT / "validation_comparison.csv",
        index=False,
    )

    with (OUTPUT / "model_selection.json").open("w") as file:
        json.dump(selections, file, indent=2)

    print("\nPARTITION COUNTS")
    print(json.dumps(metadata["partition_rows"], indent=2))

    print("\nVALIDATION COMPARISON")
    print(
        comparison[
            [
                "KPI",
                "version",
                "log_loss",
                "Brier",
                "ROC_AUC",
                "average_precision",
                "precision_top20",
                "recall_top20",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print("\nMODEL SELECTION")
    for name, selection in selections.items():
        print(
            f"{name}: {selection['retained_version']} "
            f"(C={selection['C']}, "
            f"features={selection['features']})"
        )

    print("\nPASS: households are separate.")
    print("Original test partition was not evaluated.")
    print("Outputs saved in docs/evaluation/.")
    print(
        "Top-20% results use episode IDs to break prediction ties. "
        "Constant-baseline ranking is arbitrary."
    )


if __name__ == "__main__":
    main()