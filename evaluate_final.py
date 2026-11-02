"""Final descriptive evaluation on the reused original test set.

Model choices come from validation results.
This script does not select or tune models.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from modeling import SPECS, load_data
from evaluate_models import (
    calculate_metrics,
    make_pipeline,
    save_calibration,
    save_errors,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "evaluation"
DATA_PATH = (
    ROOT / "data" / "final" / "medicaid_modeling_dataset.csv"
)


def main():
    required = [
        OUTPUT / "evaluation_metadata.json",
        OUTPUT / "model_selection.json",
        OUTPUT / "split_manifest.csv",
        ROOT / "docs" / "MODEL_SELECTION.md",
    ]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    final_path = OUTPUT / "final_test_comparison.csv"

    if final_path.exists():
        print("Final evaluation already exists:")
        print(pd.read_csv(final_path).to_string(index=False))
        print(
            "\nExisting results were displayed without refitting. "
            "Do not tune models using these results."
        )
        return

    data = load_data()

    metadata = json.loads(required[0].read_text())
    selections = json.loads(required[1].read_text())
    manifest = pd.read_csv(required[2])

    actual_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    if actual_hash != metadata["dataset_sha256"]:
        raise ValueError(
            "Dataset changed since validation. Stop and investigate."
        )

    if not manifest["episode_id"].is_unique:
        raise ValueError("Duplicate episodes in split manifest.")

    if set(manifest["episode_id"]) != set(data["episode_id"]):
        raise ValueError("Split manifest does not match dataset.")

    partition = data["episode_id"].map(
        manifest.set_index("episode_id")["partition"]
    )

    if partition.isna().any():
        raise ValueError("Some episodes have no partition.")

    train = np.flatnonzero(
        partition.isin(["development_train", "validation"])
        .to_numpy()
    )
    test = np.flatnonzero(
        partition.eq("original_test_reserved").to_numpy()
    )

    train_households = set(data.iloc[train]["household_id"])
    test_households = set(data.iloc[test]["household_id"])

    if train_households & test_households:
        raise ValueError("Training and test households overlap.")

    if len(train) != 22568 or len(test) != 7432:
        raise ValueError("Unexpected original partition sizes.")

    results = []

    for name, spec in SPECS.items():
        selection = selections[name]
        target = spec["target"]

        y_train = data.iloc[train][target]
        y_test = data.iloc[test][target]
        episode_ids = data.iloc[test]["episode_id"]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(f"{name}: both outcomes are required.")

        constant = np.full(len(test), float(y_train.mean()))

        baseline_metrics = calculate_metrics(
            y_test, constant, episode_ids
        )

        results.append(
            {
                "KPI": name,
                "version": "Constant baseline",
                "retained": False,
                **baseline_metrics,
            }
        )

        save_calibration(
            y_test,
            constant,
            f"{target}_final_baseline_calibration.csv",
        )

        revised_features = (
            list(spec["features"])
            if name == "Coverage churn"
            else ["contact_stability"]
        )
        revised_c = 0.1 if name == "Coverage churn" else 1.0

        candidates = {
            "Original": (list(spec["features"]), 1.0),
            "Revised": (revised_features, revised_c),
        }

        retained = selection["retained_version"]

        if retained not in candidates:
            raise ValueError("Invalid retained model specification.")

        expected_features, expected_c = candidates[retained]

        if (
            selection["features"] != expected_features
            or selection["C"] != expected_c
        ):
            raise ValueError(
                "Saved model selection differs from planned candidates."
            )

        for version, (features, regularization) in candidates.items():
            pipeline = make_pipeline(
                data, features, regularization
            )

            pipeline.fit(
                data.iloc[train][features],
                y_train,
            )

            probabilities = pipeline.predict_proba(
                data.iloc[test][features]
            )[:, 1]

            metrics = calculate_metrics(
                y_test, probabilities, episode_ids
            )

            results.append(
                {
                    "KPI": name,
                    "version": version,
                    "retained": version == retained,
                    **metrics,
                }
            )

            prefix = f"{target}_final_{version.lower()}"

            save_calibration(
                y_test,
                probabilities,
                f"{prefix}_calibration.csv",
            )

            save_errors(
                data,
                test,
                target,
                probabilities,
                prefix,
            )

    comparison = pd.DataFrame(results)
    comparison.to_csv(final_path, index=False)

    record = {
        "dataset_sha256": actual_hash,
        "selection_file_sha256": hashlib.sha256(
            required[1].read_bytes()
        ).hexdigest(),
        "selection_document_sha256": hashlib.sha256(
            required[3].read_bytes()
        ).hexdigest(),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "household_overlap": False,
        "selection_source": "Validation log loss only",
        "test_status": (
            "Reused original test partition; results were previously "
            "inspected in the earlier submission."
        ),
        "further_tuning": "None permitted using these test results",
        "retained_models": selections,
    }

    (OUTPUT / "final_evaluation_record.json").write_text(
        json.dumps(record, indent=2)
    )

    print("\nFINAL TEST COMPARISON")
    print(
        comparison[
            [
                "KPI",
                "version",
                "retained",
                "log_loss",
                "Brier",
                "ROC_AUC",
                "average_precision",
                "precision_top20",
                "recall_top20",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nPASS: no household overlap.")
    print("Model choices remain fixed from validation.")
    print("Results saved in docs/evaluation/.")
    print(
        "Limitation: this is a reused synthetic test set, "
        "not untouched, temporal, or external validation."
    )


if __name__ == "__main__":
    main()