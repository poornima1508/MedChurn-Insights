# MedChurn — Medicaid Renewal Analytics

A Streamlit educational application for a Renewal Operations Supervisor at a fictional state Medicaid agency.

MedChurn supports coverage-continuity review and early outreach planning using 30,000 synthetic renewal episodes.

## Live Application

[Launch MedChurn Insights](https://medchurn-insights.streamlit.app/)

Explore historical renewal outcomes, subgroup comparisons, and predictive what-if scenarios directly in your browser.

## Demonstration Video

[Watch the MedChurn demonstration](demo.mp4)



The demonstration covers the KPIs, predictive models, predictor importance, individual contributions, what-if analysis, automatically generated summaries, and interpretation limits.

If GitHub does not display a video player, download the file to watch it.

## Application Pages

| Page | Purpose |
|---|---|
| Renewal Overview | Descriptive renewal outcomes and coverage-churn statistics. |
| Subgroup Analysis | Historical synthetic outcome comparisons and filtered case review. |
| Predictive Scenarios | Model evaluation, overall importance, individual contributions, what-if comparisons, and automatic summaries. |

Subgroup comparisons describe historical synthetic outcomes. They do not evaluate predictive-model fairness.

## Decision-Maker and Objectives

The intended user is a Renewal Operations Supervisor.

The application supports:

- Prioritizing cases for supportive coverage-continuity review.
- Planning early outreach and estimating unresolved workload.
- Understanding model predictions and their limitations.
- Comparing hypothetical scenarios with a clearly identified baseline.

Financial relevance is indirect through staff-capacity planning. The project does not estimate savings, revenue, or profitability because the data contain no financial inputs or observed intervention effects.

## Key Performance Indicators

| KPI | Definition | Decision Relevance | Prediction Timeline |
|---|---|---|---|
| Coverage churn rate | Proportion of renewal episodes with a simulated coverage interruption within 30 days after renewal. | Supports continuity assistance and case-review workload planning. | Predicted 30 days before renewal. |
| Early outreach nonresponse rate | Proportion without a response by 30 days before renewal, including episodes with no outreach attempts. | Supports early contact planning and estimation of unresolved outreach workload. | Predicted 45 days before renewal, assuming household contact attributes are known then. |

Each model estimates an individual episode’s outcome probability. Averaging these probabilities over an appropriate cohort estimates its expected outcome rate.

Nonresponse is not a failure rate restricted to contacted beneficiaries. It includes episodes with no outreach attempts under the simulated historical outreach mix.

## Predictors and Modeling Rationale

### Coverage Churn Model

The model uses information assumed available 30 days before renewal.

| Predictor | Predictive Rationale |
|---|---|
| Prior coverage gaps | Historical interruptions may signal vulnerability to future coverage disruption. |
| Renewal notice sent | Indicates whether initial renewal communication was sent. |
| Renewal notice returned | Indicates a possible notice-delivery barrier. |
| Responded before prediction | Reflects recorded engagement before the prediction snapshot. |
| Prior documentation issues | Represents previous administrative barriers. |
| Administrative complexity | Summarizes administrative burden in the supplied simulation. |

These predictors reflect the supplied synthetic churn-generation process. Its relationships are hypothetical assumptions rather than findings from real Medicaid records.

Administrative complexity overlaps with some underlying indicators, which can complicate importance attribution.

### Early Outreach Nonresponse Model

This model forecasts at 45 days before renewal whether an episode will have no response by 30 days before renewal.

| Predictor | Predictive Rationale |
|---|---|
| Contact stability | Represents consistency of contact information and potential reachability. |
| Address stability | Represents a possible delivery factor whose predictive usefulness must be evaluated. |

The source does not contain timestamped household changes. The model assumes these attributes are known at the earlier prediction date.

Address stability showed little additional predictive usefulness in the fitted synthetic-data model.

Observed response, completed outreach attempts, returned notices, churn outcomes, identifiers, and protected attributes are excluded as predictors from this model.

The model forecasts under the simulated historical outreach mix. It does not estimate the effect of choosing a particular outreach intervention.

### Algorithm Selection

Both outcomes use separately fitted L2-regularized binary logistic regression models with `C = 1`.

Logistic regression was selected because it:

- Is appropriate for binary outcomes.
- Produces individual probability estimates.
- Supports exact additive explanations in log odds.
- Matches the additive log-odds structure of the supplied churn simulation.

Numeric predictors are standardized, and categorical predictors are one-hot encoded within each pipeline.

Preprocessing is fitted only on training data. No class reweighting or holdout-based hyperparameter tuning is used.

## Training and Evaluation

Training and testing use a fixed, approximately 75/25 household-grouped split.

| Setting | Value |
|---|---|
| Training episodes | 22,568 |
| Testing episodes | 7,432 |
| Random seed | 42 |
| Split unit | Household |

Members of the same household cannot appear in both partitions. Episode counts differ slightly from an exact 75/25 split because whole households are assigned together.

### Model Results

| Model | Test Prevalence | ROC-AUC | Average Precision | Brier Score | Constant-Risk Baseline Brier | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Coverage churn | 0.1350 | 0.6509 | 0.2237 | 0.1123 | 0.1167 | 0.3783 |
| Early outreach nonresponse | 0.5904 | 0.6229 | 0.6935 | 0.2270 | 0.2419 | 0.6436 |

Both models improve on their constant training-prevalence benchmarks for Brier score. Average precision exceeds test prevalence for both models.

Discrimination remains modest. Lower Brier score and log loss indicate better probability predictions. Brier score alone does not establish calibration.

These results describe an internal synthetic-data holdout. They do not establish performance on real Medicaid populations or future renewal periods.

## Overall Predictor Importance

The application displays held-out permutation importance for each model.

Importance is measured as the increase in log loss after shuffling a predictor across held-out cases, averaged over five repetitions. The standard deviation across repetitions is also displayed.

- Larger positive values indicate greater predictive usefulness.
- Near-zero or negative values provide little evidence of usefulness.
- Correlated predictors may share or mask importance.

Overall importance describes predictive usefulness across cases. It does not explain one individual prediction or establish causation.

## Individual Prediction Contributions

The application uses exact centered coefficient contributions, an appropriate alternative to SHAP for additive logistic models.

For each transformed predictor:

`Contribution = coefficient × (scenario value − mean training value)`

The reference is:

`Reference log odds = intercept + sum(coefficient × mean training value)`

The reference plus all contributions reconstructs the scenario’s model log odds. The logistic transformation converts this total to probability.

One-hot encoded components are grouped under their original predictor.

The application displays:

- Contribution direction and magnitude.
- Contributions in log odds.
- Odds multipliers relative to each predictor’s training-reference component.
- A chart distinguishing positive and negative contributions.
- Explanatory text for the selected scenario.

Positive contributions increase model log odds relative to the training reference. Negative contributions decrease them.

The training reference differs from the selected baseline episode and may not represent an actual household.

Log-odds contributions are not percentage-point changes, causal effects, or direct explanations of the difference between the baseline and updated scenario.

## What-if Analysis

For either model, users can:

1. Select a prediction model.
2. Select a held-out baseline renewal episode.
3. Inspect the baseline inputs and predicted risk.
4. Modify predictor values.
5. Run the what-if analysis.
6. Compare baseline and updated predictions.
7. Review individual contributions and an automatic summary.
8. Download the scenario summary.

The selected baseline remains fixed while scenario inputs change.

Unchanged inputs and unchanged predictions are valid results. A modified input does not have to produce a different prediction.

The application rejects a returned notice without a sent notice. Invalid submissions clear previous results to prevent stale predictions from being displayed.

Input values are bounded to observed values, but combinations may still be uncommon.

All results are model-based scenarios, not guaranteed effects of real-world interventions.

## Automatically Generated Summaries

Every valid scenario submission generates a deterministic summary containing:

- All submitted scenario inputs.
- Changed inputs, or an explicit statement that none changed.
- Baseline predicted risk.
- Updated predicted risk.
- Change from baseline in percentage points.
- Practical implications for supportive review or outreach planning.
- Relevant assumptions and limitations.

Differences are calculated using unrounded predictions.

Summaries require no external language model or API key and can be downloaded as text files.

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

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch the application:

```bash
python -m streamlit run main.py
```

Open the local URL printed by Streamlit.

The first visit to Predictive Scenarios trains both models. Later visits reuse models cached in memory.

No credentials, downloaded model files, or external data services are required to run the application locally.

Dependencies are recorded in `requirements.txt`. Dependency checks passed in the tested macOS environment. Installation on other operating systems has not been independently verified.

## Verification

### Model Checks

```bash
python evaluate.py
```

This saves:

- `docs/model_metrics.json`
- `docs/importance_churn_30d.csv`
- `docs/importance_nonresponse.csv`

It verifies:

- Household separation between training and testing.
- Exact reconstruction of individual prediction log odds.
- Consistent predictions for unchanged inputs.

These checks passed in the tested macOS environment.

### Application Checks

```bash
python test_app.py
```

Automated application checks passed for:

- All three pages loading without displayed errors.
- Unchanged scenarios for both models.
- Modified scenarios for both models.
- Required automatic-summary content.
- Individual contribution tables.
- Invalid notice input rejection.
- Removal of stale results after invalid input.
- Recovery after correcting invalid inputs.

The downloaded scenario summary was separately opened and checked. The contribution chart was visually compared with its table values.

A missing `ScriptRunContext` warning may appear during automated testing and does not indicate a failed check.

### Dependency Checks

```bash
python -m pip check
```

The tested environment reported:

```text
No broken requirements found.
```

## Project Files

| File or Folder | Purpose |
|---|---|
| `main.py` | Application entry point and navigation. |
| `Renewal_Overview.py` | Descriptive renewal dashboard. |
| `modeling.py` | Model training, evaluation, explanations, and summaries. |
| `pages/` | Subgroup analysis and predictive scenario pages. |
| `data/raw/` | Synthetic source tables. |
| `data/final/` | Final synthetic datasets used by the application. |
| `code/` | Data-generation and validation scripts. |
| `requirements.txt` | Recorded dependency versions. |
| `evaluate.py` | Model evaluation and consistency checks. |
| `test_app.py` | Automated application checks. |
| `docs/PROGRESS_REPORT.md` | KPI justification, predictor rationale, results, accomplishments, and limitations. |
| `docs/CONTRIBUTIONS.md` | Member contributions and AI-assistance disclosure. |
| `docs/DATA_SHARING.md` | Data provenance and sharing information. |
| `docs/model_metrics.json` | Saved model evaluation results. |
| `docs/importance_churn_30d.csv` | Churn-model permutation importance. |
| `docs/importance_nonresponse.csv` | Nonresponse-model permutation importance. |
| `demo.mp4` | Application demonstration recording. |

Run data-generation scripts only from the project directory. Regenerating data may overwrite included CSV files.

## Progress and Contributions

Completed work includes:

- Descriptive renewal and subgroup pages.
- Two predictive modeling pipelines.
- Household-separated holdout evaluation.
- Global predictor importance and individual prediction explanations.
- Editable what-if scenarios with fixed baselines.
- Automatically generated, downloadable summaries.
- Model and application verification scripts.
- Setup instructions and supporting documentation.
- Deployment of the live Streamlit application.

See [the progress report](docs/PROGRESS_REPORT.md) for details.

See [member contributions](docs/CONTRIBUTIONS.md) for each contributor’s completed work and the AI-assistance disclosure.

## Data Provenance and Sharing

All supplied beneficiary, household, renewal, outreach, administrative, and churn records are synthetic.

No actual Medicaid administrative records or real beneficiary identifiers were used. No restricted external dataset is required to run the application.

Included generation scripts and raw/final CSV files support reproduction of the educational project.

See [data provenance and sharing](docs/DATA_SHARING.md) for details.

Do not upload virtual environments, credentials, API keys, private documents, or real beneficiary data.

## Interpretation and Limitations

- Predictions support educational planning scenarios.
- Models do not determine eligibility or justify denial of services.
- Predictive importance and contributions do not establish causation.
- Churn is simulated rather than an observed enrollment event.
- Early nonresponse includes episodes with no outreach attempts.
- Contact attributes are assumed known at the earlier prediction time.
- Some scenario combinations may be uncommon.
- Financial savings and intervention effects are not estimated.
- Excluding protected attributes does not establish model fairness.
- Subgroup comparisons describe historical synthetic outcomes.
- Real-world use requires authorized data, timestamp verification, external and temporal evaluation, calibration assessment, subgroup model evaluation, and human review.

## Final Submission Checks

Before submitting, confirm that:

- The demonstration link opens the uploaded `demo.mp4`.
- The recording meets the requested seven-minute duration and demonstrates both models.
- Audio is clear and application content is visible.
- Both members have verified their contribution statements.
- The instructor and all members can access the repository and recording.

Submit this repository link through the course portal:

https://github.com/poornima1508/MedChurn-Insights
