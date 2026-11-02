# Model Selection and Validation Findings

## Evaluation Procedure

Model choices were made using the household-separated validation
partition of 4,487 episodes. Development training contained 18,081
episodes.

The original 7,432-episode test partition was excluded from these
comparisons. Its results were previously inspected in the earlier
submission, so it is a reused evaluation set, not a new untouched test.

The planned selection rule was lower validation log loss.

## Validation Comparison

| KPI | Constant Baseline Log Loss | Original Log Loss | Revised Log Loss | Retained Version |
|---|---:|---:|---:|---|
| Coverage churn | 0.392148 | 0.370583 | 0.370560 | Revised: C = 0.1 |
| Early outreach nonresponse | 0.685385 | 0.657313 | 0.657197 | Revised: Contact Stability only |

Both revised models have slightly lower validation log loss and
Brier score than their original versions.

The differences between original and revised models are very small.
They do not establish a substantial or statistically reliable
performance improvement.

The churn revision slightly reduces ROC-AUC and average precision.
The nonresponse revision increases ROC-AUC but reduces average
precision. These trade-offs are retained in the report.

The revised models are selected under the predefined log-loss rule.
The simpler nonresponse model also removes a predictor with little
demonstrated usefulness.

No additional model candidates will be selected using final-test
results.

## Validation Error Findings

### Coverage Churn

High-complexity cases have the highest segment Brier score, 0.189978.
Their average predicted risk is 26.36%, compared with an observed
outcome rate of 26.83%.

Higher Brier scores across segments must be interpreted alongside
their outcome rates. They do not independently establish worse
calibration or unfairness.

For Low Contact Stability, average predicted churn is 14.32% versus
an observed rate of 16.17%, indicating underprediction on average
in this validation segment.

Some episodes assigned approximately 6.45% churn risk nevertheless
have a churn outcome. A low-risk estimate cannot rule out an event.

Users should not exclude low-risk cases from assistance solely
because of a prediction.

### Early Outreach Nonresponse

For High Contact Stability, predicted nonresponse is approximately
50.03%, compared with an observed rate of 47.89%. Individual outcomes
remain uncertain.

For Low Contact Stability, predicted nonresponse is approximately
83.34%, compared with an observed rate of 80.64%. Some cases in this
category still respond.

The revised model gives identical probabilities to cases with the
same Contact Stability category. It cannot reliably rank people
within a category.

The top-20% review calculation breaks probability ties using episode
IDs. That tie-breaking rule has no substantive decision meaning.

Administrative Complexity and coverage-history segments are used
only for retrospective diagnostics in this nonresponse analysis.
They are not inputs to the earlier nonresponse forecast.

## Decision-Support Implications

Use probabilities for illustrative workload planning and supportive
review, not guaranteed individual outcomes.

Do not interpret scenario changes as causal intervention effects.
Do not use predictions to decide eligibility or justify denial.

## Final Evaluation

Refit both original and revised specifications on the full original
training partition and report their results against the constant
baseline on the reserved original test partition.

Keep the validation-selected models regardless of which specification
performs better on that reused test partition.

Report test diagnostics without further tuning.