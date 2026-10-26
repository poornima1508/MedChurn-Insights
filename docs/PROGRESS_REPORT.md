# MedChurn — Progress Report

Poornima Anamaneni Sayeeswaran and Md Arshad Khan  
M.S. Analytics | Masters Research Project  
October 26, 2026

## Decision-Maker and Objective

MedChurn supports a Renewal Operations Supervisor at a fictional state Medicaid agency. The application helps prioritize supportive case review and plan outreach resources, with coverage continuity and effective use of staff capacity as organizational objectives.

The project uses 30,000 synthetic renewal episodes. These records do not represent actual Medicaid beneficiaries. Predictions support educational planning scenarios and must not determine eligibility or justify denial of services.

## 1. Key Performance Indicators and Justification

| KPI | Definition and Denominator | Relevance to the Decision-Maker |
|---|---|---|
| Coverage churn rate | Number of episodes with churn_30d = 1 divided by all renewal episodes. The outcome represents a simulated coverage interruption within 30 days after renewal. | Supports prioritizing coverage-continuity assistance and estimating potential case-review workload. Not all churn is necessarily procedural or preventable. |
| Early outreach nonresponse rate | Number of episodes with responded_pre_prediction = 0 divided by all renewal episodes, including episodes with no outreach attempts. The response window extends from 45 to 30 days before renewal. | Supports early contact planning and estimation of unresolved outreach workload under the simulated historical outreach mix. |

Each model predicts an individual episode’s probability of its outcome. Averaging these probabilities over an appropriate cohort estimates the expected cohort outcome rate.

Financial relevance is indirect: workload planning may support efficient use of staff capacity. The dataset contains no staff costs, revenue, healthcare utilization, or observed intervention effects, so financial savings and profitability cannot be estimated.

## 2. Predictors and Predictive Models

### Coverage Churn Model

The model predicts churn 30 days before renewal using information available by that snapshot.

| Predictor | Predictive Relevance |
|---|---|
| Prior coverage gaps | Historical interruptions may indicate vulnerability to future coverage disruption. |
| Renewal notice sent | Indicates whether initial renewal communication was sent. |
| Renewal notice returned | Indicates a possible notice-delivery barrier. |
| Responded before prediction | Reflects recorded engagement before the prediction snapshot. |
| Prior documentation issues | Represents previous administrative barriers. |
| Administrative complexity | Summarizes administrative burden in the supplied simulation. |

These predictors correspond to the supplied synthetic churn-generation process. Its coefficients are hypothetical assumptions rather than estimates from real Medicaid records.

Administrative complexity overlaps with underlying administrative indicators, so predictive importance may be shared. Neither predictor importance nor coefficients establish causation.

### Early Outreach Nonresponse Model

This model forecasts, at 45 days before renewal, whether an episode will have no response by 30 days before renewal.

| Predictor | Predictive Relevance |
|---|---|
| Contact stability | Represents consistency of contact information and potential reachability. |
| Address stability | Represents a possible communication-delivery factor whose usefulness must be evaluated. |

The synthetic response generator uses contact stability and randomly generated outreach attempts. Address stability is not directly used in that generator and showed almost no additional predictive importance in the fitted model.

The source has no timestamped household changes. The model assumes these household attributes were known 45 days before renewal. This timing assumption requires verification with real records.

Observed response, completed outreach attempts, returned notices, churn outcomes, identifiers, and protected attributes are excluded as predictors from this model.

The model forecasts under the simulated historical outreach-attempt mix. It does not estimate response to a selected intervention or number of attempts. At 30 days before renewal, its outcome is already known; it must therefore be interpreted as a forecast from the earlier date.

### Algorithm Selection

Both models use L2-regularized binary logistic regression with C = 1.

Logistic regression was selected because it:

- Is appropriate for binary outcomes.
- Produces individual probability estimates.
- Supports exact additive explanations in log odds.
- Matches the additive log-odds structure of the synthetic churn generator.

Each outcome has a separately fitted model and predictor set. No class reweighting is used.

Numeric predictors are standardized, and categorical predictors are one-hot encoded. Preprocessing is fitted only on training data within each modeling pipeline.

### Evaluation Method

A fixed household-separated holdout uses approximately 75% of episodes for training and 25% for testing, with random seed 42.

The realized split contains:

- 22,568 training episodes.
- 7,432 testing episodes.

Whole households remain in one partition, preventing household overlap between training and testing. No hyperparameter tuning was performed on the holdout.

Models are trained from the included CSV when the predictive page first loads and are cached in memory.

### Verified Model Results

| Model | Test Prevalence | ROC-AUC | Average Precision | Brier Score | Constant-Risk Baseline Brier | Log Loss |
|---|---:|---:|---:|---:|---:|---:|
| Coverage churn | 0.1350 | 0.6509 | 0.2237 | 0.1123 | 0.1167 | 0.3783 |
| Early outreach nonresponse | 0.5904 | 0.6229 | 0.6935 | 0.2270 | 0.2419 | 0.6436 |

For both models, average precision exceeds test prevalence, and Brier score improves on the constant training-prevalence benchmark.

Discrimination remains modest. Brier score measures overall probability error; it does not establish calibration by itself.

These results describe an internal synthetic-data holdout. No confidence intervals, temporal validation, external validation, real-population calibration, or predictive fairness assessment are claimed.

## 3. Expanded Application Functionality

### Overall Predictor Importance

The application displays held-out permutation importance for each model.

Importance is the mean increase in log loss after shuffling a predictor across held-out episodes, using five repetitions. The standard deviation across repetitions is displayed.

Larger positive values indicate greater predictive usefulness. Near-zero or negative values provide little evidence of usefulness.

For churn, administrative complexity and response before prediction were the most important predictors in the displayed results. For early nonresponse, contact stability dominated address stability.

Overall importance describes predictive usefulness across cases, rather than explaining one individual prediction.

### Individual Predictor Contributions

Individual explanations use exact centered coefficient contributions, an appropriate alternative to SHAP for additive logistic models.

For each transformed predictor:

Contribution = coefficient × (scenario value − mean training value)

Reference log odds are calculated as:

Reference = intercept + sum(coefficient × mean training value)

The reference plus all contributions reconstructs the scenario’s model log odds. The logistic transformation converts that total to predicted probability.

One-hot encoded contributions are combined under their original predictor.

The training reference may not represent an actual household. It is distinct from the selected episode baseline used in the what-if comparison.

### Contextual Interpretation

The application explains the direction and magnitude of each contribution:

- Positive contributions increase model log odds relative to the training reference.
- Negative contributions decrease model log odds relative to the training reference.
- Exponentiating a contribution expresses its odds multiplier relative to that reference component.

Log-odds contributions are not percentage-point changes. They explain a scenario relative to the training reference, not its difference from the selected baseline episode.

Readable predictor labels, signed contributions, odds multipliers, and chart tooltips support interpretation. Positive and negative contributions use different colors.

These results describe predictive associations. They do not establish causation or demonstrate the effect of an intervention. Correlated and derived predictors may complicate attribution.

### What-if Analysis

For each model, users can:

1. Select a held-out baseline episode.
2. Inspect baseline inputs and predicted risk.
3. Modify predictor values.
4. Run the scenario.
5. Compare baseline and updated risk.
6. Inspect individual contributions.

The selected baseline remains fixed. The application reports the difference in percentage points, including zero change. Differences use unrounded predictions.

Returned notices without a sent notice are rejected. Invalid submissions clear previous scenario results so stale predictions are not displayed.

Inputs are restricted to observed ranges or categories, although combinations may still be uncommon.

These are illustrative model-based scenarios rather than guaranteed intervention effects.

### Automatically Generated Summaries

Every valid scenario run generates a deterministic natural-language summary containing:

- All scenario inputs.
- Changed fields.
- Baseline and updated predictions.
- Difference from baseline, including no change.
- Practical implications.
- Relevant limitations.

Summaries can be downloaded as text files. No external language model, API key, or patient-data transfer is required.

## 4. Implementation and Verification Progress

Completed work includes:

- Preserved descriptive renewal and subgroup analysis pages.
- Repaired application navigation.
- Built two separate predictive pipelines.
- Added held-out model evaluation.
- Added overall importance and individual explanations.
- Added baseline comparisons and editable scenario inputs.
- Added automatic downloadable summaries.
- Improved predictor labels and categorical input handling.
- Corrected disability-flag handling to accept Yes/No and numeric representations.
- Updated charts to display explicit values and readable labels.
- Retained synthetic data, generation scripts, and previous validation outputs.
- Recorded dependency versions and updated setup instructions.

The application and automated checks were run on macOS using Python 3.14.6.

### Model Verification

The evaluation script passed checks for:

- Household isolation between training and testing.
- Exact reconstruction of individual prediction log odds.
- Prediction consistency for unchanged inputs.

### Application Verification

Automated application checks passed for:

- All three pages loading without displayed errors.
- Unchanged scenarios for both models.
- Modified scenarios for both models.
- Required automatic-summary content.
- Individual contribution tables.
- Invalid notice input rejection.
- Removal of stale results following invalid input.
- Recovery after correcting invalid inputs.

The downloaded scenario summary was opened and checked for the required content. The corrected contribution chart was visually checked against its table values.

Dependency checks reported no broken requirements. Installation on other operating systems has not been verified.

### Verified Scenario Examples

| Model | Input Change | Baseline Risk | Updated Risk | Difference |
|---|---|---:|---:|---:|
| Coverage churn | No changes | 16.17% | 16.17% | 0.00 percentage points |
| Coverage churn | Responded Before Prediction: 0 → 1 | 16.17% | 11.10% | −5.07 percentage points |
| Coverage churn | Renewal Notice Returned: 0 → 1 | 16.17% | 24.62% | +8.44 percentage points |
| Early outreach nonresponse | No changes | 65.50% | 65.50% | 0.00 percentage points |
| Early outreach nonresponse | Contact Stability: Medium → High | 65.50% | 49.54% | −15.96 percentage points |

Displayed differences use unrounded predictions and may differ slightly from subtraction of the rounded risks.

These examples confirm application behavior for selected scenarios. They do not show that real-world interventions would produce the displayed differences.

## 5. Limitations and Future Work

The main limitations are:

- Simulation assumptions drive the data and results.
- Churn is a sampled synthetic label rather than an observed enrollment event.
- Nonresponse combines contacted and uncontacted episodes.
- Earlier household attributes are assumed to be available and static.
- Some input combinations may be uncommon.
- Model discrimination is modest.
- Excluding protected attributes does not establish fairness.
- Historical subgroup comparisons do not measure model fairness.
- Financial savings and intervention effects cannot be estimated.

Operational use would require authorized real data, timestamp verification, temporal and external validation, calibration assessment, subgroup error analysis, staffing and cost inputs, and human review.

## 6. Remaining Deliverables

Before submission:

- Confirm that each contributor’s statement reflects completed work.
- Record the seven-minute demonstration.
- Replay the recording to verify audio, readability, and unobstructed screen content.
- Upload the application, authorized data, documentation, and recording or video link.
- Confirm instructor and collaborator access to all materials.
- Submit the repository or folder link.

## 7. Contributor Statement

Individual responsibilities are documented in docs/CONTRIBUTIONS.md and must accurately reflect work completed by each contributor.

ChatGPT assisted with implementation, troubleshooting, and documentation drafts. Contributors are responsible for reviewing the implementation, understanding the methods, verifying results, and disclosing assistance according to course requirements.