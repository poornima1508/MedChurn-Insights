# MedChurn Insights — Medicaid Renewal Analytics

A Streamlit educational application supporting coverage-continuity
review and early outreach planning for a Renewal Operations Supervisor
at a fictional state Medicaid agency.

The project uses 30,000 synthetic renewal episodes. Predictions must
not determine eligibility or justify denial of services.

## Application and Documentation

- [Live application](https://medchurn-insights.streamlit.app/)
- [Week 11 demonstration video](https://github.com/poornima1508/MedChurn-Insights/blob/main/week11_demo.mp4)
- [Evaluation report](docs/EVALUATION_REPORT.md)
- [User-testing findings](docs/USER_TESTING.md)
- [Evaluation plan](docs/EVALUATION_PLAN.md)
- [Model-selection record](docs/MODEL_SELECTION.md)
- [Member contributions and AI disclosure](docs/CONTRIBUTIONS.md)
- [Data provenance and sharing](docs/DATA_SHARING.md)
- [Preserved Week 10 submission](https://github.com/poornima1508/MedChurn-Insights/tree/week10-submission)

## Week 11 Demonstration

[Watch or download week11_demo.mp4](https://github.com/poornima1508/MedChurn-Insights/blob/main/week11_demo.mp4)

The demonstration accompanies the revised application and evaluation
report. If GitHub does not display a video player, download the MP4
using the file page's download control.

The recording should be reviewed for clear audio, readable application
content, duration, and access before submission.

### Archived Week 10 Recording

[Watch the Week 10 demonstration](https://sluedu-my.sharepoint.com/personal/poornima_anamanenisayeeswaran_slu_edu/_layouts/15/stream.aspx?id=%2Fpersonal%2Fpoornima%5Fanamanenisayeeswaran%5Fslu%5Fedu%2FDocuments%2FRecordings%2FCall%20with%20Md%20Arshad%20Khan%2D20261026%5F223317%2DMeeting%20Recording%2Emp4)

The archived recording is hosted on university SharePoint and may
require an authorized account. It demonstrates the previous submission.

## Application Pages

| Page | Purpose |
|---|---|
| Renewal Overview | Descriptive renewal outcomes and coverage-churn statistics |
| Subgroup Analysis | Historical synthetic outcome comparisons and filtered case review |
| Predictive Scenarios | Model performance, explanations, validated scenarios, comparisons, and automatic summaries |

Descriptive subgroup comparisons do not establish predictive-model
fairness.

## Decision-Maker and Objectives

The intended user is a Renewal Operations Supervisor.

The application supports:

- Prioritizing supportive coverage-continuity review.
- Planning early outreach and anticipated workload.
- Understanding predictions and their limitations.
- Comparing alternative case assumptions against a fixed baseline.

Financial relevance is indirect through resource planning.
Costs, savings, revenue, and profitability are not estimated.

## Retained Key Performance Indicators

Both Week 10 KPIs are retained.

| KPI | Definition and denominator | Decision relevance | Prediction timeline |
|---|---|---|---|
| Coverage churn rate | Episodes with churn_30d=1 divided by all renewal episodes; simulated interruption within 30 days after renewal | Continuity assistance and case-review planning | 30 days before renewal |
| Early outreach nonresponse rate | Episodes with responded_pre_prediction=0 divided by all renewal episodes, including those with no outreach attempts | Early outreach and anticipated nonresponse workload planning | Forecast at 45 days before renewal for nonresponse by 30 days before renewal |

Individual probabilities may be averaged over an appropriate cohort
to estimate its expected outcome rate.

Nonresponse includes uncontacted cases under the simulated historical
outreach mix. It is not a failure rate restricted to contacted cases.

## Predictive Models

Both models use L2-regularized logistic regression with numeric
standardization and one-hot encoding of categorical predictors.
Preprocessing is fitted only on training data.

Logistic regression supports binary probability prediction and exact
additive explanations in log odds. Its structure also matches the
additive assumptions in the synthetic churn generator.

### Coverage Churn

| Predictor | Rationale |
|---|---|
| Prior Coverage Gaps | Historical interruptions may indicate vulnerability to disruption |
| Renewal Notice Sent | Indicates whether initial renewal communication was sent |
| Renewal Notice Returned | Indicates a possible delivery barrier |
| Responded Before Prediction | Describes engagement recorded before the snapshot |
| Prior Documentation Issues | Represents previous administrative barriers |
| Administrative Complexity | Summarizes administrative burden in the simulation |

Original model: C=1.0.

Retained revised model: C=0.1, testing stronger regularization with
the same predictors.

Administrative Complexity is derived from documentation issues,
income verification requirements, and whether prior gaps exceed zero.
Overlapping predictors can complicate importance attribution.

### Early Outreach Nonresponse

Original model: C=1.0 using Contact Stability and Address Stability.

Retained revised model: C=1.0 using Contact Stability only.

Contact Stability describes contact-information consistency and
potential reachability. Removing Address Stability tests a simpler model.

Observed response and completed outreach attempts are excluded because
they occur during the forecast window.

Cases within the same Contact Stability category receive the same
probability. The retained model cannot distinguish cases within
that category.

## Evaluation Design

We reproduced the original household-grouped development/test split
using seed 42, then divided the development partition using seed 43.

| Partition | Episodes | Purpose |
|---|---:|---|
| Development training | 18,081 | Fit candidate models |
| Validation | 4,487 | Compare candidates and select versions |
| Original test | 7,432 | Evaluate fixed model choices |
| Total | 30,000 | Complete synthetic dataset |

No household appears in multiple partitions.

Candidates used identical partitions and evaluation metrics.
Model versions were selected using validation log loss, then refitted
on the combined 22,568 development episodes.

The original test results were already inspected in Week 10.
This is a reused synthetic test set, not an untouched final holdout.
Model choices remained fixed after final evaluation.

This design does not establish performance on future periods or
real Medicaid populations.

## Leakage and Timing Review

- Churn inputs are assumed available 30 days before renewal.
- The nonresponse forecast starts 45 days before renewal.
- The recorded prediction_date refers to the churn snapshot at day −30.
- Contact Stability is assumed known at the earlier forecast origin.
- Outcomes, identifiers, and protected attributes are excluded as predictors.
- Preprocessing is fitted on training data only.
- Household grouping separates related household members across partitions.

Individual response and returned-notice timestamps and complete
administrative feature snapshots are unavailable. Source-code review
supports the intended synthetic timeline but cannot establish
real-world point-in-time availability.

## Final Evaluation Results

The constant baseline assigns training outcome prevalence to every case.

| KPI | Version | Log loss | Brier score | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|
| Coverage churn | Constant baseline | 0.395729 | 0.116750 | 0.500000 | 0.134957 |
| Coverage churn | Original | 0.378251 | 0.112341 | 0.650880 | 0.223712 |
| Coverage churn | Revised — retained | 0.378250 | 0.112342 | 0.650829 | 0.223601 |
| Early outreach nonresponse | Constant baseline | 0.676937 | 0.241937 | 0.500000 | 0.590420 |
| Early outreach nonresponse | Original | 0.643575 | 0.226969 | 0.622932 | 0.693472 |
| Early outreach nonresponse | Revised — retained | 0.643559 | 0.226960 | 0.623543 | 0.679335 |

Lower log loss and Brier score indicate less probability error.
ROC-AUC measures ranking ability. Average precision summarizes
positive-outcome ranking and should be interpreted alongside prevalence.

Retained-model log loss is approximately 4.4% lower than the constant
baseline for churn and 4.9% lower for nonresponse.

Both revisions were selected for slightly lower validation log loss.
Improvements over the original models are tiny and do not establish
statistically significant superiority.

The churn revision slightly worsens test Brier score and ranking
metrics. The nonresponse revision slightly improves probability error
but lowers average precision and top-20% yield.

### Illustrative Review Capacity

| Retained model | Outcome rate in top 20% | Share of outcomes in top 20% |
|---|---:|---:|
| Coverage churn | 23.87% | 35.39% |
| Early outreach nonresponse | 78.75% | 26.69% |

The 20% capacity is illustrative, not a staffing recommendation.
Prediction ties are resolved by episode ID, which has no substantive
decision meaning. This particularly affects the nonresponse model,
which produces only three distinct probabilities.

## Error Analysis and Calibration

The application displays probability-band calibration tables and
segment-level error diagnostics.

Findings include:

- Low predicted churn does not rule out a churn outcome.
- High-complexity and prior-gap segments had higher validation Brier scores.
- Each Contact Stability category contains both responders and nonresponders.
- Small probability bands provide unstable calibration estimates.

Higher segment Brier scores can reflect outcome prevalence as well as
prediction error. They do not independently prove miscalibration
or unfairness.

For nonresponse, later administrative fields used in retrospective
diagnostics are not forecasting inputs.

No confidence intervals or formal significance tests were calculated.

See [the evaluation report](docs/EVALUATION_REPORT.md) for details.

## Predictor Importance and Individual Contributions

### Overall Importance

Permutation importance measures the mean increase in log loss after
shuffling a predictor across reused test cases, using five repetitions.

Larger positive values suggest greater predictive usefulness.
Near-zero or negative values show little evidence of usefulness.
Correlated predictors can share or mask importance.

### Individual Contributions

Individual explanations use centered coefficient contributions:

`Contribution = coefficient × (transformed value − mean training value)`

The reference log odds plus all contributions reconstruct the
prediction's log odds. Encoded components are grouped under their
original predictor.

The application displays predictor values, log-odds contributions,
odds multipliers, charts, and written explanations.

Positive contributions indicate higher risk relative to the training
reference; negative contributions indicate lower risk.

The training reference differs from the selected baseline episode.
Contributions are not percentage-point effects, causal effects,
or direct attributions of the baseline-to-scenario difference.

## What-if Analysis

For either model:

1. Select a prediction model.
2. Select a held-out baseline episode.
3. Inspect baseline inputs, risk, and explanation.
4. Review fixed, operational, external, and derived input roles.
5. Modify permitted scenario assumptions.
6. Select Run What-if Analysis.
7. Compare inputs and predictions side by side.
8. Review and download the automatic summary.

The selected baseline remains fixed. Unchanged predictions are valid,
including when an input changes.

### Input Checks

- Historical gaps and prior documentation issues are fixed.
- Numeric inputs use observed ranges.
- Categories are restricted to permitted values.
- Administrative Complexity is recalculated.
- A returned notice requires a sent notice.
- A response requires at least one outreach attempt.
- Invalid submissions clear previous scenario results.
- Uncommon profiles may trigger warnings.

Passing validation does not establish real-world plausibility.

Responses, returned mail, and verification requirements are not freely
controlled actions. Edits compare alternative conditions at the same
snapshot, not guaranteed intervention outcomes.

## Automatic Summaries

Each valid run generates a deterministic summary containing:

- Scenario inputs and changed values.
- Baseline and updated predictions.
- Difference in percentage points.
- Practical implications and trade-offs.
- Input cautions and limitations.

Differences use unrounded probabilities.
Summaries require no external language model or API key.

## User Testing and Improvement

Tarun (P1), a classmate outside the project group with no Medicaid
experience, completed a guided walkthrough on November 2, 2026.

The facilitator explained the concept and supplied two scenario
instructions. Tarun's reported interpretation identified the direction
of risk changes, a possible case-review use, and prediction uncertainty.

He suggested displaying actual input values beside contributions.
This was implemented in tables, chart labels, tooltips, and written
explanations for both models.

Automated checks passed after the update.

Independent factor identification and participant-created scenarios
were not fully documented. Exposure to AI-generated practice answers
was not recorded. This provides preliminary feedback rather than
evidence of operational effectiveness.

See [the user-testing record](docs/USER_TESTING.md).

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

Install dependencies and check consistency:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Launch:

```bash
python -m streamlit run main.py
```

Open the local URL printed by Streamlit.

The first visit to Predictive Scenarios fits the selected models.
Later visits reuse models cached in memory.

Keep the included docs/evaluation/ files: the application uses them
to verify selections and display evaluation results.

No API keys, downloaded model files, or external data services
are required locally.

Installation in a fresh virtual environment and dependency checks
passed on the tested Mac. Other operating systems have not been
independently verified.

## Verification

### Application Checks

```bash
python test_app.py
```

Checks passed for:

- Prediction consistency and explanation reconstruction.
- Fixed history and recalculated complexity.
- Numeric and categorical validation.
- All three application pages.
- Unchanged and modified scenarios for both models.
- Fixed baseline predictions.
- Required summaries and contribution tables.
- Invalid notice and response combinations.
- Clearing stale results and recovery after correction.

Live screenshots confirmed both model scenario workflows and
the predictor-value display.

A missing ScriptRunContext warning may appear during automated
testing and does not indicate a failed check.

### Evaluation Workflow

```bash
python evaluate_models.py
python evaluate_final.py
```

evaluate_models.py compares candidates on validation data and saves
selection records.

evaluate_final.py compares fixed versions on the reused original
test partition. Its existing-results guard avoids repeating evaluation
when the final comparison is already saved.

Results are included in docs/evaluation/. Running these scripts is
not required for normal application launch.

Do not repeatedly change models in response to final-test results.
Regenerating evaluation records can invalidate saved selection checks.

### Historical Evaluation

evaluate.py and the original evaluation outputs directly under docs/
are retained for Week 10 traceability.

Use docs/evaluation/final_test_comparison.csv and the Week 11 report
for current comparisons.

## Project Files

| File or folder | Purpose |
|---|---|
| main.py | Entry point and navigation |
| Renewal_Overview.py | Descriptive dashboard |
| modeling.py | Original modeling utilities, data loading, and exact explanations |
| decision_support.py | Selected models, scenario checks, comparisons, and summaries |
| pages/ | Subgroup and predictive scenario pages |
| evaluate_models.py | Development/validation comparison and selection |
| evaluate_final.py | Fixed-model final comparison |
| evaluate.py | Historical Week 10 evaluation |
| test_app.py | Updated application checks |
| requirements.txt | Recorded dependencies |
| data/raw/ | Synthetic source tables |
| data/final/ | Synthetic application datasets |
| code/ | Generation and validation scripts |
| docs/EVALUATION_REPORT.md | Week 11 evaluation report |
| docs/EVALUATION_PLAN.md | Evaluation design |
| docs/MODEL_SELECTION.md | Selection rationale |
| docs/USER_TESTING.md | Participant findings and improvement |
| docs/evaluation/ | Metrics, split manifest, metadata, calibration, and errors |
| docs/PROGRESS_REPORT.md | Historical Week 10 progress report |
| docs/CONTRIBUTIONS.md | Actual contributions and AI disclosure |
| docs/DATA_SHARING.md | Provenance and sharing information |
| week11_demo.mp4 | Week 11 demonstration recording |

Generation scripts use paths relative to the project directory.
Regenerating data may overwrite CSVs and invalidate evaluation hashes.

## Data Provenance and Sharing

All supplied beneficiary, household, renewal, outreach, administrative,
and churn records are synthetic.

No actual Medicaid administrative records or real beneficiary
identifiers were identified in the supplied project. No restricted
external dataset is required.

Raw and final CSVs and generation scripts support reproduction.
See [data-sharing documentation](docs/DATA_SHARING.md).

Do not upload virtual environments, caches, credentials, private
documents, or real beneficiary data.

## Limitations and Future Priorities

- Synthetic outcomes reflect hypothetical generation assumptions.
- Matching those assumptions does not establish real-world validity.
- The original test partition was reused.
- Evaluation was neither temporal nor external.
- Point-in-time predictor availability is incompletely verified.
- Nonresponse includes cases with no outreach attempts.
- Model revisions produced only tiny improvements.
- Ranking remains modest, with many nonresponse ties.
- Excluding protected attributes does not establish fairness.
- Scenario comparisons do not establish causation.
- Staffing costs and intervention effectiveness are unknown.
- User testing involved one guided substitute participant.

Future priorities include authorized timestamped data, temporal or
external evaluation, calibration and subgroup error assessment,
uncertainty estimates, staffing inputs, and testing with renewal staff.

## Traceability and Contributions

The [week10-submission tag](https://github.com/poornima1508/MedChurn-Insights/tree/week10-submission)
preserves the identifiable Week 10 submission.

Subsequent commits document Week 11 evaluation and application changes.

See [member contributions](docs/CONTRIBUTIONS.md) for completed work
and AI-assistance disclosure.

## Final Submission Checks

- Revised application and evaluation outputs are included.
- Setup instructions and dependencies are documented.
- Evaluation and guided user-testing reports are included.
- The Week 11 MP4 is linked above.
- Week 10 is preserved with a Git tag.
- User-testing evidence limitations are disclosed.

Before submission, verify video playback, audio, readable visuals,
recording duration, final contribution statements, and instructor
and group-member access.

Submit the repository link through the course portal.