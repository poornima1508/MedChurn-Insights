# Model Evaluation Plan

## Objectives

Retain the existing Coverage churn and Early outreach nonresponse
KPIs. Assess predictive performance, prediction errors, explanation
quality, scenario validity, and usefulness to the intended user.

## Preservation

The previous submission is preserved in the GitHub tag
week10-submission.

Original data-generation scripts and supplied datasets will not
be overwritten during model evaluation.

## Evaluation Design

Reproduce the original household-grouped 75/25 split using seed 42.

Keep the original test households outside model development.
Split the original training households into approximately 80%
development-training and 20% validation households using seed 43.

Use identical partitions for the original model, revised model,
and simple baseline for each KPI.

Fit preprocessing only on the relevant training partition.

Select the retained model using validation results. Then refit it
on the full original training partition and evaluate it on the
original test partition.

The original test results were already inspected in the previous
submission. Therefore, this is a reused evaluation set, not a new,
untouched final test. No further tuning will follow its results.

Household separation supports evaluation on unseen households.
It does not establish performance on future renewal periods.
No temporal or external validation is claimed.

## Baselines and Metrics

The simple probability baseline assigns every case the outcome
rate calculated from training data.

Primary selection metric: log loss, where lower is better.

Supporting metrics:
- Brier score: overall probability prediction error.
- ROC-AUC: ability to rank cases by outcome risk.
- Average precision: ranking performance, interpreted alongside
  outcome prevalence.
- Calibration tables: predicted probabilities versus observed rates.

For workload planning, also report precision and recall among the
highest-risk 20% of cases. This is an illustrative review capacity,
not a validated operational staffing policy.

## Planned Model Improvements

Coverage churn:
Compare the original logistic regression model with C = 1 against
stronger regularization with C = 0.1, keeping the predictors unchanged.
This tests whether shrinking coefficients improves validation
probability estimates when predictors overlap.

Early outreach nonresponse:
Compare the original two-predictor logistic regression model against
a model using Contact Stability only.
Address Stability is not used directly in the response generator;
removing it may reduce unnecessary noise and simplify explanations.

Retain the revised model only if its validation log loss is lower.
Otherwise retain the original model. Report supporting metrics and
trade-offs even when the attempted improvement is unsuccessful.

Do not add further candidates based on test results.

## Prediction Timing and Leakage Checks

Coverage churn:
- Prediction snapshot: 30 days before renewal.
- Outcome window ends 30 days after renewal.
- Exclude outcome labels, identifiers, and outcome-window information
  from predictors.
- Prior history must refer to information recorded before prediction.
- Notices are dated 45 days before renewal.
- Returned notices, responses, and administrative requirements lack
  individual event timestamps. Their availability by prediction time
  is an assumption requiring verification with real records.

Early outreach nonresponse:
- Forecast origin: 45 days before renewal.
- Outcome: no response by 30 days before renewal.
- Exclude observed response and completed outreach attempts.
- Contact and address stability are assumed known at day -45.
- The supplied prediction_date represents day -30, not the earlier
  nonresponse forecast origin.

Administrative Complexity is derived from documentation issues,
income verification requirements, and prior coverage gaps.
This is predictor overlap, not outcome leakage by itself.

## Error Analysis

Inspect performance and probability errors by:
- Contact Stability.
- Administrative Complexity.
- Prior Coverage Gaps: zero versus one or more.

Report segment sizes, observed outcome rates, average predicted
risks, and Brier scores. Flag small segments and avoid treating
descriptive differences as definitive evidence.

Use development and validation findings to guide interpretation.
Final-test diagnostics will be reported without further tuning.

## Application Validation

- Distinguish fixed history, external conditions, and potentially
  changeable operational inputs.
- Recalculate derived complexity from its underlying inputs.
- Reject returned notices without a sent notice.
- Check response and outreach-attempt consistency.
- Display baseline and scenario inputs and predictions side by side.
- Explain uncertainty, trade-offs, and noncausal interpretation.

## Outside-Participant Test

Recruit at least one person outside the project group.

Ask the participant to interpret a baseline prediction, identify
its main contributing factors, compare two plausible scenarios,
describe a possible decision, and identify unclear information.

Record actual feedback and misunderstandings. Implement at least
one resulting improvement or explain why no change is warranted.

Document any lack of Medicaid operations experience.
Do not invent participant feedback or expose restricted data.