# MedChurn — Medicaid Renewal Analytics

A Streamlit educational application for a Renewal Operations Supervisor at a fictional state Medicaid agency.

The application supports coverage-continuity review and early outreach planning using synthetic data.

## Setup and Launch

The application and automated checks were run on macOS using Python 3.14.6.

From the project directory, create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Open the local URL printed by Streamlit.

The first visit to Predictive Scenarios trains both models. Later visits reuse models cached in memory. No API keys, downloaded models, or external services are required.

Dependencies are recorded in requirements.txt. Dependency checks passed in the tested macOS environment. Installation on other operating systems has not been verified.

## Application Pages

- **Renewal Overview:** Descriptive renewal outcomes and coverage-churn statistics.
- **Subgroup Analysis:** Historical synthetic outcome comparisons and filtered case review.
- **Predictive Scenarios:** Model performance, overall predictor importance, individual contributions, what-if comparisons, and automatic summaries.

Subgroup comparisons describe historical outcomes; they do not evaluate predictive-model fairness.

## Key Performance Indicators

| KPI | Definition | Prediction Timeline |
|---|---|---|
| Coverage churn rate | Proportion of renewal episodes with a simulated coverage interruption within 30 days after renewal. | Predicted 30 days before renewal. |
| Early outreach nonresponse rate | Proportion without a response by 30 days before renewal, including episodes with no outreach attempts. | Predicted 45 days before renewal, assuming household contact attributes are known then. |

Individual probabilities can be averaged over an appropriate cohort to estimate its expected outcome rate.

Coverage churn risk supports prioritizing continuity assistance. Nonresponse risk supports early outreach and workload planning. Neither model estimates financial savings.

## Predictive Models

Both models use logistic regression with preprocessing fitted only on training data.

### Coverage Churn Predictors

- Prior coverage gaps.
- Renewal notice sent.
- Renewal notice returned.
- Response before the prediction date.
- Prior documentation issues.
- Administrative complexity.

### Early Outreach Nonresponse Predictors

- Contact stability.
- Address stability.

The nonresponse model excludes completed outreach attempts and observed response status as predictors. It forecasts under the simulated historical outreach mix.

Logistic regression is suitable for binary outcomes, produces probability estimates, and supports exact additive explanations in log odds.

See [the progress report](docs/PROGRESS_REPORT.md) for individual predictor justifications and modeling assumptions.

## Training and Evaluation

Training and testing are separated by household using a fixed, approximately 75/25 split:

- Training: 22,568 episodes.
- Testing: 7,432 episodes.
- Random seed: 42.

Members of the same household cannot appear in both sets. Preprocessing is fitted only on training data, and the holdout is not used for tuning.

| Model | ROC-AUC | Average Precision | Brier Score | Constant-Risk Baseline Brier |
|---|---:|---:|---:|---:|
| Coverage churn | 0.6509 | 0.2237 | 0.1123 | 0.1167 |
| Early outreach nonresponse | 0.6229 | 0.6935 | 0.2270 | 0.2419 |

Lower Brier scores indicate better probability predictions. Both models improve on their constant-risk benchmarks, but discrimination remains modest.

These results describe an internal synthetic-data holdout, not real-world or temporal validation.

## Predictor Importance and Individual Contributions

**Overall importance** measures predictive usefulness across held-out cases using permutation importance: the increase in log loss after shuffling a predictor.

**Individual contributions** explain one scenario using exact additive logistic contributions relative to mean training features.

Positive contributions increase model log odds relative to the training reference; negative contributions decrease them.

The training reference differs from the selected baseline episode used for what-if comparisons. Log-odds contributions are not percentage-point changes or evidence of causation.

Charts display readable predictor names and hover values. Individual contribution charts distinguish positive and negative contributions by color.

## What-if Analysis

Users can:

1. Select a prediction model.
2. Select a baseline renewal episode.
3. Inspect the baseline prediction and inputs.
4. Modify predictor values.
5. Run the scenario.
6. Compare baseline and updated predictions.
7. Review individual contributions and an automatic summary.

Unchanged inputs and unchanged predictions are valid results.

A returned notice without a sent notice is rejected. Invalid submissions clear previous scenario results so that stale predictions are not displayed.

Every valid scenario run generates a summary containing:

- Scenario inputs.
- Changed inputs.
- Baseline and updated predictions.
- Difference from the baseline.
- Practical implications.
- Relevant limitations.

Summaries can be downloaded as text files. Differences are calculated using unrounded predictions.

Results are model-based scenarios, not guaranteed effects of real-world interventions.

## Verification

### Model Checks

```bash
python evaluate.py
```

This saves:

- docs/model_metrics.json
- docs/importance_churn_30d.csv
- docs/importance_nonresponse.csv

It verifies:

- Household separation between training and testing.
- Exact reconstruction of individual prediction log odds.
- Consistent predictions for unchanged inputs.

These checks passed in the Python 3.14.6 macOS environment.

### Application Checks

```bash
python test_app.py
```

Automated application checks passed for:

- All three pages loading without displayed errors.
- Unchanged scenarios for both predictive models.
- Modified scenarios for both predictive models.
- Required automatic-summary content.
- Individual contribution tables.
- Invalid notice input rejection.
- Removal of stale results after invalid input.
- Recovery after correcting invalid inputs.

The downloaded scenario summary was separately opened and checked for the required content. The updated contribution chart was visually checked against its table values.

The missing ScriptRunContext warning may appear during automated testing and does not indicate a failed check.

### Dependency Checks

```bash
python -m pip check
```

The verified environment reported:

```text
No broken requirements found.
```

## Project Files

| File or Folder | Purpose |
|---|---|
| main.py | Application entry point and navigation. |
| Renewal_Overview.py | Historical renewal dashboard. |
| modeling.py | Model training, evaluation, explanations, and summaries. |
| pages/ | Subgroup analysis and predictive scenario pages. |
| data/ | Synthetic raw and final CSV files. |
| code/ | Data-generation and validation scripts. |
| requirements.txt | Recorded dependency versions. |
| evaluate.py | Model evaluation and consistency checks. |
| test_app.py | Automated application checks. |
| docs/PROGRESS_REPORT.md | KPI justification, predictor rationale, model results, accomplishments, and limitations. |
| docs/CONTRIBUTIONS.md | Contributor responsibilities and AI-assistance disclosure. |
| docs/DATA_SHARING.md | Data provenance, sharing scope, and interpretation limits. |
| docs/model_metrics.json | Reproducible model evaluation results. |
| docs/importance_churn_30d.csv | Overall churn-model predictor importance. |
| docs/importance_nonresponse.csv | Overall nonresponse-model predictor importance. |

Historical documentation may remain in the project. Use this README for current setup instructions.

Run generation scripts only from the project directory. Regenerating data may overwrite the supplied CSV files.

## Demonstration Video

The seven-minute demonstration recording will be added or linked before submission.

The recording will demonstrate:

- KPI definitions and decision relevance.
- Predictors and algorithm rationale.
- Both predictive models.
- Overall importance and individual contributions.
- Baseline and modified scenarios.
- Automatically generated summaries.
- Interpretation and limitations.

Replay the recording to confirm clear audio, readable visuals, and unobstructed application content.

If the recording is hosted separately, verify its viewing permissions independently.

## Data and Interpretation Limits

All supplied beneficiary and household records are synthetic.

- Predictions do not determine eligibility or justify denial of services.
- Predictive importance and contributions do not establish causation.
- Churn is simulated rather than an observed enrollment event.
- Early nonresponse includes cases with no outreach attempts.
- Some scenario combinations may be uncommon.
- Financial savings are not estimated.
- Excluding protected attributes does not establish model fairness.
- Real-world use requires authorized data, timestamp verification, external validation, calibration assessment, and subgroup model evaluation.

Review [the data-sharing documentation](docs/DATA_SHARING.md) and confirm authorization before publishing project materials.

## Submission Checklist

- [ ] Updated application and authorized data included.
- [ ] Setup instructions and dependencies reviewed.
- [ ] Progress report finalized.
- [ ] Each contributor's actual work confirmed and documented.
- [ ] Seven-minute demonstration video added or linked.
- [ ] Video playback and audio checked.
- [ ] Instructor and all collaborators have access.
- [ ] Repository or folder link submitted through the course portal.

Do not upload virtual environments, caches, or credentials.