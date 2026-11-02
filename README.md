# MedChurn Insights — Medicaid Renewal Analytics

A Streamlit educational application supporting coverage-continuity review and early outreach planning for a Renewal Operations Supervisor at a fictional state Medicaid agency.

The project uses 30,000 synthetic renewal episodes. Predictions must not determine eligibility or justify denial of services.

## Application and Documentation

- [Live application](https://medchurn-insights.streamlit.app/)
- [Week 11 evaluation report](docs/EVALUATION_REPORT.md)
- [User-testing findings](docs/USER_TESTING.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [Model-selection record](docs/MODEL_SELECTION.md)
- [Member contributions and AI-assistance disclosure](docs/CONTRIBUTIONS.md)
- [Data provenance and sharing](docs/DATA_SHARING.md)
- [Preserved Week 10 submission](https://github.com/poornima1508/MedChurn-Insights/tree/week10-submission)

The live application reflects the version deployed from GitHub. Local changes must be pushed and deployed before they appear online.

## Demonstration Recordings

### Week 11 — Pending

A new seven-minute demonstration will cover:

- Evaluation design and leakage checks.
- Baseline, original, and revised model results.
- Model-selection decisions and limitations.
- Individual explanations and overall predictor importance.
- A complete baseline-to-scenario comparison.
- Input validation and automatic summaries.
- User-testing feedback and the resulting display improvement.

The Week 11 recording link will be added before submission.

### Week 10 — Archived Recording

[Watch the Week 10 demonstration](https://sluedu-my.sharepoint.com/personal/poornima_anamanenisayeeswaran_slu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fpoornima%5Fanamanenisayeeswaran%5Fslu%5Fedu%2FDocuments%2FRecordings%2FCall%20with%20Md%20Arshad%20Khan%2D20261026%5F223317%2DMeeting%20Recording%2Emp4)

This recording demonstrates the previous submission. It does not replace the Week 11 demonstration.

The recording is hosted on university SharePoint through Microsoft Stream. Viewing may require an authorized university account. Instructor and group-member viewing permissions must be verified separately.

## Application Pages

| Page | Purpose |
|---|---|
| Renewal Overview | Descriptive renewal outcomes and coverage-churn statistics |
| Subgroup Analysis | Historical synthetic outcome comparisons and filtered case review |
| Predictive Scenarios | Evaluation results, explanations, validated scenarios, side-by-side comparisons, and automatic summaries |

Descriptive subgroup comparisons do not establish predictive-model fairness.

## Decision-Maker and Objectives

The intended user is a Renewal Operations Supervisor.

The application supports:

- Prioritizing cases for supportive continuity review.
- Planning early outreach and anticipated workload.
- Understanding model predictions and their limitations.
- Comparing alternative case assumptions against a fixed baseline.

Financial relevance is indirect through resource planning. The project does not estimate costs, savings, revenue, or profitability.

## Retained Key Performance Indicators

Both Week 10 KPIs are retained.

| KPI | Definition and denominator | Decision relevance | Prediction timeline |
|---|---|---|---|
| Coverage churn rate | Episodes with churn_30d=1 divided by all renewal episodes; simulated interruption within 30 days after renewal | Continuity assistance and case-review planning | 30 days before renewal |
| Early outreach nonresponse rate | Episodes with responded_pre_prediction=0 divided by all renewal episodes, including those with no outreach attempts | Early outreach and anticipated nonresponse workload planning | Forecast at 45 days before renewal for nonresponse by 30 days before renewal |

Individual probabilities may be averaged over an appropriate cohort to estimate its expected outcome rate.

Nonresponse is not a failure rate restricted to contacted beneficiaries. It includes uncontacted cases under the simulated historical outreach mix.

## Predictive Models

Both models use L2-regularized logistic regression with numeric standardization and one-hot encoding of categorical predictors.

Preprocessing is fitted only on training data.

Logistic regression supports binary probability prediction and exact additive explanations in log odds. Its structure also matches the additive log-odds assumptions used in the synthetic churn generator.

### Coverage Churn

| Predictor | Rationale |
|---|---|
| Prior Coverage Gaps | Historical interruptions may indicate vulnerability to disruption |
| Renewal Notice Sent | Describes whether initial renewal communication was sent |
| Renewal Notice Returned | Describes a possible delivery barrier |
| Responded Before Prediction | Describes engagement recorded before the snapshot |
| Prior Documentation Issues | Represents previous administrative barriers |
| Administrative Complexity | Summarizes administrative burden in the simulation |

Original model: C=1.0.

Retained revised model: C=0.1, testing stronger regularization with the same predictors.

Administrative Complexity is derived from documentation issues, income verification requirements, and whether prior coverage gaps exceed zero. Overlapping inputs can complicate importance attribution.

### Early Outreach Nonresponse

Original model: C=1.0 using Contact Stability and Address Stability.

Retained revised model: C=1.0 using Contact Stability only.

Contact Stability describes a household's contact-information consistency and potential reachability. Removing Address Stability tests a simpler model.

Observed response and completed outreach attempts are excluded as predictors because they occur during the forecast window.

Cases in the same Contact Stability category receive the same probability. The retained model cannot distinguish cases within that category.

## Evaluation Design

We reproduced the original household-grouped development/test split using seed 42, then divided the original development partition using seed 43.

| Partition | Episodes | Purpose |
|---|---:|---|
| Development training | 18,081 | Fit candidate models |
| Validation | 4,487 | Compare candidates and select versions |
| Original test | 7,432 | Evaluate fixed model choices |
| Total | 30,000 | Complete synthetic dataset |

No household appears in multiple partitions.

Both candidates for each KPI used identical partitions and metrics. Model versions were selected using validation log loss, then refitted on the combined 22,568 development episodes.

The original test results were already inspected during Week 10. Week 11 therefore uses a reused synthetic test set, not an untouched final holdout.

Model choices remained fixed after final evaluation. This design does not establish performance on future periods or real Medicaid populations.

See [the evaluation report](docs/EVALUATION_REPORT.md) for validation results, final comparisons, error analysis, and limitations.

## Leakage and Timing Checks

- Churn predictors are assumed available 30 days before renewal.
- The nonresponse forecast starts 45 days before renewal.
- The dataset's recorded prediction_date is the churn snapshot at day −30.
- Contact Stability is assumed known at the earlier nonresponse forecast origin.
- Outcomes, identifiers, and protected attributes are excluded as predictors.
- Training-fitted preprocessing prevents evaluation-data information from entering preprocessing estimates.
- Household grouping prevents related household members from appearing across partitions.

Individual response and returned-notice timestamps and complete administrative feature snapshots are unavailable. Source-code review supports the intended synthetic timeline but cannot establish real-world point-in-time availability.

## Final Evaluation Results

The constant baseline assigns the training outcome prevalence to every case.

| KPI | Version | Log loss | Brier score | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|
| Coverage churn | Constant baseline | 0.395729 | 0.116750 | 0.500000 | 0.134957 |
| Coverage churn | Original | 0.378251 | 0.112341 | 0.650880 | 0.223712 |
| Coverage churn | Revised — retained | 0.378250 | 0.112342 | 0.650829 | 0.223601 |
| Early outreach nonresponse | Constant baseline | 0.676937 | 0.241937 | 0.500000 | 0.590420 |
| Early outreach nonresponse | Original | 0.643575 | 0.226969 | 0.622932 | 0.693472 |
| Early outreach nonresponse | Revised — retained | 0.643559 | 0.226960 | 0.623543 | 0.679335 |

Lower log loss and Brier score indicate less probability error. ROC-AUC measures ranking ability. Average precision summarizes positive-outcome ranking and should be interpreted alongside prevalence.

The retained models reduce log loss relative to the constant baseline by approximately 4.4% for churn and 4.9% for nonresponse.

Both revisions were selected for slightly lower validation log loss. Improvements over the original models are tiny and do not establish statistically significant superiority.

The churn revision slightly worsens test Brier score and ranking metrics. The nonresponse revision slightly improves probability error but lowers average precision and top-20% yield.

### Illustrative Review Capacity

| Retained model | Outcome rate in top 20% | Share of outcomes in top 20% |
|---|---:|---:|
| Coverage churn | 23.87% | 35.39% |
| Early outreach nonresponse | 78.75% | 26.69% |

The 20% review capacity is illustrative, not a staffing recommendation.

Equal-risk cases are ordered by episode ID, which has no substantive decision meaning. This particularly affects the nonresponse model because it produces only three distinct probabilities.

## Error Analysis and Calibration

The application displays calibration tables and segment-level diagnostics.

Key findings include:

- Low predicted churn does not rule out a churn outcome.
- High-complexity and prior-gap segments had higher validation Brier scores.
- Contact Stability groups still contain both responders and nonresponders.
- Small probability bands provide unstable calibration estimates.

Higher segment Brier scores can reflect different outcome prevalence as well as prediction error. They do not independently establish miscalibration or unfairness.

For nonresponse, later administrative fields used in retrospective diagnostics are not forecasting inputs.

No confidence intervals or formal significance tests were calculated.

## Overall Importance and Individual Contributions

### Overall Predictor Importance

Permutation importance measures the mean increase in log loss after shuffling a predictor across reused test cases, using five repetitions.

- Larger positive values indicate greater predictive usefulness.
- Near-zero or negative values provide little evidence of usefulness.
- Correlated predictors may share or mask importance.

This describes patterns across cases, not one prediction.

### Individual Contributions

Individual explanations use exact centered coefficient contributions:

`Contribution = coefficient × (transformed value − mean training value)`

The reference log odds plus all contributions reconstruct the prediction's log odds. Encoded components are grouped under their original predictor.

The application shows:

- Predictor names and actual input values.
- Contributions in log odds.
- Odds multipliers.
- Charts with predictor values.
- Written explanations of direction and magnitude.

Positive contributions indicate higher model risk relative to the training reference; negative contributions indicate lower model risk.

The training reference differs from the selected baseline episode. Contributions are not percentage-point effects, causal effects, or direct attributions of the baseline-to-scenario difference.

## What-if Analysis

For either model:

1. Select a prediction model.
2. Select a held-out baseline episode.
3. Inspect the baseline prediction, inputs, and explanation.
4. Review which inputs are fixed, external, operational, or derived.
5. Modify permitted scenario assumptions.
6. Select Run What-if Analysis.
7. Compare baseline and scenario inputs and predictions side by side.
8. Review and download the automatic summary.

The baseline remains fixed while scenario assumptions change.

Unchanged inputs and unchanged predictions are valid results. A modified input does not have to change the prediction.

### Input Checks

- Historical coverage gaps and documentation issues are fixed.
- Numeric inputs use observed ranges.
- Categorical inputs use permitted values.
- Administrative Complexity is recalculated from underlying inputs.
- A returned notice requires a sent notice.
- A recorded response requires at least one outreach attempt.
- Invalid submissions clear previous results.
- Uncommon profiles may trigger warnings.

Passing validation does not establish real-world plausibility.

Responses, returned mail, and verification requirements are not freely controlled actions. Scenario edits compare alternative conditions at the same snapshot, not guaranteed intervention outcomes.

## Automatic Summaries

Every valid submission generates a deterministic summary containing:

- Scenario inputs and changed values.
- Baseline and updated predictions.
- Difference in percentage points.
- Practical implications.
- Relevant trade-offs.
- Input cautions and limitations.

Differences use unrounded probabilities.

Summaries require no external language model or API key and can be downloaded as text.

## User Testing and Resulting Improvement

One student outside the project group, P1, completed a guided walkthrough on November 2, 2026.

P1 had no Medicaid experience. The facilitator explained the concept and supplied two scenario instructions.

The participant's reported interpretation identified the direction of both risk changes, a possible case-review use, and that predictions are uncertain.

P1 suggested showing predictor values beside individual contributions. The application now includes those values in tables, chart labels, tooltips, and written explanations.

Automated application checks passed after the update.

This was one assisted session with a substitute participant. Exposure to example answers was not recorded, and independent factor identification was not separately established. A participant retest of the revised display has not been documented.

See [the full user-testing record](docs/USER_TESTING.md).

## Setup and Launch

Verified locally on macOS using Python 3.14.6.

From the project directory:

```bash
python3 -m venv .venv
```

Activate on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Check dependency consistency:

```bash
python -m pip check
```

Launch:

```bash
python -m streamlit run main.py
```

Open the local URL printed by Streamlit.

The first visit to Predictive Scenarios fits the selected models using the saved evaluation configuration. Later visits reuse models cached in memory.

Keep the included docs/evaluation/ files: the application uses them to verify the saved selections and display evaluation results.

No API keys, downloaded model files, or external data services are required locally.

A fresh virtual environment was created and the recorded requirements installed successfully on the tested Mac. Other operating systems have not been independently verified.

## Verification and Evaluation Scripts

### Application Checks

```bash
python test_app.py
```

The updated checks passed after the predictor-value display change, covering:

- Prediction consistency and explanation reconstruction.
- Fixed history and recalculated complexity.
- Numeric and categorical validation.
- Loading all three pages.
- Unchanged and modified scenarios for both models.
- Fixed baseline predictions.
- Required summaries and contribution tables.
- Invalid notice and response combinations.
- Clearing stale results and recovery after correction.

A missing ScriptRunContext warning may appear during AppTest execution. It does not indicate a failed check.

### Evaluation Workflow

```bash
python evaluate_models.py
python evaluate_final.py
```

evaluate_models.py compares candidates on validation data and writes selection records.

evaluate_final.py evaluates the fixed choices on the reused original test partition. Its existing-results guard avoids repeating evaluation when the final comparison is already saved.

Saved results are included in docs/evaluation/. These scripts are not required for normal application launch.

Do not repeatedly change model choices in response to final-test results. Regenerating evaluation records can also invalidate the saved selection checks used by the application.

### Historical Week 10 Evaluation

evaluate.py and the original top-level evaluation outputs in docs/ are retained for traceability. They describe the original models, not the complete Week 11 selection workflow.

Use docs/evaluation/final_test_comparison.csv and the Week 11 evaluation report for current comparisons.

## Project Files

| File or folder | Purpose |
|---|---|
| main.py | Application entry point and navigation |
| Renewal_Overview.py | Descriptive renewal dashboard |
| modeling.py | Original modeling utilities, data loading, and exact explanations |
| decision_support.py | Selected-model fitting, scenario checks, comparisons, and summaries |
| pages/ | Subgroup analysis and predictive scenario pages |
| evaluate_models.py | Development/validation comparison and model selection |
| evaluate_final.py | Fixed-model comparison on the reused original test set |
| evaluate.py | Historical Week 10 evaluation |
| test_app.py | Updated automated application checks |
| requirements.txt | Recorded dependency versions |
| data/raw/ | Synthetic source tables |
| data/final/ | Synthetic application datasets |
| code/ | Data-generation and validation scripts |
| docs/EVALUATION_REPORT.md | Week 11 evaluation and decision-support report |
| docs/EVALUATION_PLAN.md | Evaluation design and comparison procedure |
| docs/MODEL_SELECTION.md | Selection decisions and rationale |
| docs/USER_TESTING.md | Guided participant test and resulting improvement |
| docs/evaluation/ | Split manifest, metadata, comparisons, calibration, and error outputs |
| docs/PROGRESS_REPORT.md | Historical Week 10 progress report |
| docs/CONTRIBUTIONS.md | Actual member contributions and AI disclosure |
| docs/DATA_SHARING.md | Data provenance and sharing information |

Generation scripts use paths relative to the project directory. Regenerating data may overwrite the included CSV files and invalidate recorded evaluation hashes.

## Data Provenance and Sharing

All supplied beneficiary, household, renewal, outreach, administrative, and churn records are synthetic.

No actual Medicaid administrative records or real beneficiary identifiers were identified in the supplied project. No restricted external dataset is required to run the application.

Raw and final CSVs and generation scripts are included for coursework reproduction.

See [data-sharing documentation](docs/DATA_SHARING.md). Members should confirm authorization to share group-owned materials.

Do not upload virtual environments, caches, credentials, private documents, or real beneficiary data.

## Remaining Limitations

- Synthetic labels reflect hypothetical generation assumptions.
- Matching the generator does not establish real-world validity.
- The original test partition was reused.
- Evaluation was neither temporal nor external.
- Point-in-time feature availability is not fully verified.
- Nonresponse includes cases with no outreach attempts.
- Contact attributes are assumed known at the earlier forecast origin.
- Model revisions produced only tiny improvements.
- Ranking is modest, with many nonresponse prediction ties.
- Excluding protected attributes does not establish fairness.
- Scenario differences do not establish causation.
- Operational costs and intervention effectiveness are unknown.
- One guided student test provides limited usability evidence.

Future priorities include authorized timestamped data, temporal or external evaluation, calibration assessment, subgroup error analysis, uncertainty estimates, staffing inputs, and testing with renewal operations staff.

## Traceability and Contributions

The identifiable Week 10 submission is preserved by the
[week10-submission tag](https://github.com/poornima1508/MedChurn-Insights/tree/week10-submission).

Subsequent commits document the Week 11 evaluation and application changes.

See [member contributions](docs/CONTRIBUTIONS.md) for actual completed work and AI-assistance disclosure.

## Week 11 Submission Checklist

- [x] Revised application implemented and local checks passed.
- [x] Original and revised models compared with a constant baseline.
- [x] Evaluation design, timing assumptions, and limitations documented.
- [x] Guided outside-participant test documented.
- [x] Predictor-value display improvement implemented.
- [x] Week 10 submission preserved with an identifiable tag.
- [ ] Confirm updated member contributions.
- [ ] Add the new seven-minute Week 11 demonstration link.
- [ ] Verify recording audio, playback, and viewing permissions.
- [ ] Push the final changes and verify the deployed application.
- [ ] Verify instructor and group-member access.
- [ ] Submit the updated repository link through the course portal.