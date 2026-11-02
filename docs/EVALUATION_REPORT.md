# MedChurn Insights — Model Evaluation and Decision Support

## 1. Purpose and Retained KPIs

The intended decision-maker is a Renewal Operations Supervisor at a
fictional state Medicaid agency. The application supports supportive
case review and outreach planning, with coverage continuity as the
organizational objective.

We retained both Week 10 KPIs:

| KPI | Definition | Intended use |
|---|---|---|
| Coverage churn rate | Proportion of renewal episodes with churn_30d=1, representing a simulated coverage interruption within 30 days after renewal | Plan continuity assistance and case-review workload |
| Early outreach nonresponse rate | Proportion with responded_pre_prediction=0 by 30 days before renewal, including episodes with no outreach attempts | Plan early outreach resources and anticipated nonresponse workload |

Individual predicted probabilities may be averaged over an appropriate
cohort to estimate its expected outcome rate.

All 30,000 episodes are synthetic. Predictions must not determine
eligibility or justify denial of services. Neither model estimates
financial savings or causal intervention effects.

## 2. Evaluation Design

### Data partitions

We reproduced the Week 10 household-grouped split using random seed 42.
The original development partition was then divided by household using
seed 43.

| Partition | Episodes | Purpose |
|---|---:|---|
| Development training | 18,081 | Fit original and revised candidate models |
| Validation | 4,487 | Compare candidates and select model versions |
| Original test | 7,432 | Evaluate the fixed selections |
| Total | 30,000 | Complete synthetic dataset |

No household appears in more than one partition. Grouping limits
information sharing between related household members across partitions.

This design evaluates generalization to held-out households within the
same synthetic population. It does not evaluate future calendar periods
or real Medicaid populations.

Preprocessing was fitted only on the applicable training partition.
Candidate comparisons used identical partitions and evaluation metrics.

### Separation of development and final evaluation

We selected model versions using validation log loss. After selection,
each version was refitted on the combined development-training and
validation partitions, totaling 22,568 episodes.

The final comparison evaluated the constant baseline, original model,
and revised model on the same 7,432 original test episodes. Model
choices remained fixed.

The original test results had already been inspected during Week 10.
This is therefore a reused synthetic test set, not an untouched final
holdout. We did not use the Week 11 final comparison to change model
choices.

Partition assignments, selection records, and evaluation metadata are
stored in docs/evaluation/.

## 3. Data-Leakage and Prediction-Timing Review

### Coverage churn

The prediction snapshot is 30 days before renewal. Predictors are:

- Prior coverage gaps.
- Renewal notice sent.
- Renewal notice returned.
- Response before prediction.
- Prior documentation issues.
- Administrative complexity.

The generation scripts describe notices and early outreach as occurring
before the prediction snapshot. Historical gaps and documentation issues
are treated as prior information.

Administrative complexity is calculated from prior documentation issues,
income verification required, and whether prior coverage gaps exceed zero.
Overlap between a derived predictor and its components is not itself
target leakage, although it can complicate importance interpretation.

The outcome window ends 30 days after renewal. Outcome labels and
post-outcome information are excluded from predictors.

### Early outreach nonresponse

The forecast origin is 45 days before renewal. The target describes
nonresponse by 30 days before renewal.

The original model used Contact Stability and Address Stability.
The retained revised model uses Contact Stability only.

Observed response status and completed outreach attempts are excluded
from this model's predictors because they belong to the forecast window.

The dataset's recorded prediction_date refers to the churn snapshot,
30 days before renewal. The earlier nonresponse forecast origin is a
modeling assumption, not a separately recorded feature snapshot.

### Remaining timing uncertainty

Contact and address attributes are assumed known and static at the
earlier forecast origin. Individual response and returned-notice event
timestamps are unavailable. Administrative predictor timestamps are
also incomplete.

Consequently, source-code review supports the intended synthetic
timeline, but does not establish real-world point-in-time availability.
Operational use would require timestamped feature snapshots and
verification of outcome availability before training.

Identifiers, protected attributes, and outcome dates are not model
predictors. All preprocessing is trained without evaluation-partition
data.

## 4. Metrics and Simple Baseline

The simple baseline assigns every episode the outcome prevalence
estimated from the applicable training partition.

| Metric | Interpretation and relevance |
|---|---|
| Log loss | Primary selection metric; evaluates probability predictions and penalizes confidently incorrect predictions |
| Brier score | Mean squared probability error; lower values indicate better probability predictions |
| ROC-AUC | Measures ranking ability across outcome classes |
| Average precision | Summarizes positive-outcome ranking; interpreted alongside outcome prevalence |
| Precision in top 20% | Outcome rate among cases ranked in an illustrative review-capacity group |
| Recall in top 20% | Share of all outcomes captured in that illustrative group |
| Calibration tables | Compare mean predictions with observed rates within probability bands |

The top-20% capacity is illustrative, not a staffing recommendation.
Prediction ties are resolved by episode ID, which has no substantive
decision meaning. The constant baseline cannot meaningfully rank cases.

## 5. Attempted Model Improvements

Both original models use logistic regression with training-fitted
preprocessing. Logistic regression supports binary probability prediction
and exact additive explanations in log odds.

| KPI | Original model | Proposed revision | Rationale |
|---|---|---|---|
| Coverage churn | Logistic regression, C=1.0, six predictors | Same predictors, C=0.1 | Test whether stronger regularization improves probability estimates |
| Early outreach nonresponse | Logistic regression, C=1.0, Contact Stability and Address Stability | Logistic regression, C=1.0, Contact Stability only | Test a simpler model that removes a predictor without a direct role in the response-generation mechanism |

### Validation comparison

| KPI | Version | Log loss | Brier score | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|
| Coverage churn | Constant baseline | 0.392148 | 0.115349 | 0.500000 | 0.133051 |
| Coverage churn | Original | 0.370583 | 0.109549 | 0.662783 | 0.247843 |
| Coverage churn | Revised | 0.370560 | 0.109545 | 0.662315 | 0.247129 |
| Early outreach nonresponse | Constant baseline | 0.685385 | 0.246122 | 0.500000 | 0.564965 |
| Early outreach nonresponse | Original | 0.657313 | 0.232935 | 0.609836 | 0.658628 |
| Early outreach nonresponse | Revised | 0.657197 | 0.232880 | 0.616068 | 0.647532 |

Both revised versions were retained because they had lower validation
log loss under the predefined selection procedure.

The improvements were very small. We do not claim statistically
established superiority or a major accuracy improvement. The simpler
nonresponse model also sacrifices some ranking performance.

## 6. Final Evaluation Results

These results use the reused original test partition.

| KPI | Version | Log loss | Brier score | ROC-AUC | Average precision |
|---|---|---:|---:|---:|---:|
| Coverage churn | Constant baseline | 0.395729 | 0.116750 | 0.500000 | 0.134957 |
| Coverage churn | Original | 0.378251 | 0.112341 | 0.650880 | 0.223712 |
| Coverage churn | Revised — retained | 0.378250 | 0.112342 | 0.650829 | 0.223601 |
| Early outreach nonresponse | Constant baseline | 0.676937 | 0.241937 | 0.500000 | 0.590420 |
| Early outreach nonresponse | Original | 0.643575 | 0.226969 | 0.622932 | 0.693472 |
| Early outreach nonresponse | Revised — retained | 0.643559 | 0.226960 | 0.623543 | 0.679335 |

Compared with the constant baseline, retained-model log loss is lower
by approximately 4.4% for churn and 4.9% for nonresponse.

Both models provide predictive information beyond a constant estimate
within this synthetic evaluation. Discrimination remains modest, and
these results do not establish operational benefit.

The churn revision has nearly identical log loss and slightly worse
Brier score and ranking metrics than the original. The nonresponse
revision has slightly better probability error and ROC-AUC, but lower
average precision. We retained the validation-selected versions rather
than changing choices after seeing test results.

### Illustrative review-capacity results

| KPI | Version | Outcome rate in top 20% | Share of outcomes in top 20% |
|---|---|---:|---:|
| Coverage churn | Original | 23.87% | 35.39% |
| Coverage churn | Revised — retained | 23.87% | 35.39% |
| Early outreach nonresponse | Original | 79.09% | 26.80% |
| Early outreach nonresponse | Revised — retained | 78.75% | 26.69% |

The retained churn model concentrates more churn outcomes in the
illustrative review group than an uninformative ranking would generally
provide. However, it still leaves many outcomes outside that group.

The nonresponse model produces only three distinct probabilities,
one per Contact Stability category. It cannot distinguish cases within
the same category, so top-20% results depend partly on arbitrary tie
resolution.

## 7. Prediction Errors and Calibration

Validation error analysis examined Contact Stability, administrative
complexity, and prior coverage-gap segments.

### Coverage churn

In validation:

- High-complexity cases had a Brier score of approximately 0.1900,
  compared with 0.0779 for low-complexity cases.
- Cases with prior gaps had a Brier score of approximately 0.1601,
  compared with 0.0899 for cases without gaps.
- Low Contact Stability cases had an observed churn rate of 16.17%
  versus a mean prediction of 14.32%.
- Some cases with predicted churn risk of approximately 6.45%
  nevertheless had a simulated churn outcome.

A low prediction does not rule out churn. Users should not exclude
cases from assistance solely because predicted risk is low.

### Early outreach nonresponse

In validation:

- High Contact Stability cases had observed nonresponse of 47.89%
  versus a mean prediction of 50.03%.
- Low Contact Stability cases had observed nonresponse of 80.64%
  versus a mean prediction of 83.34%.
- Some low-stability cases responded despite a predicted nonresponse
  probability of approximately 83.34%.

The model describes category-level patterns and cannot predict each
person's behavior reliably.

### Interpretation limits

Higher segment Brier scores can reflect different outcome prevalence
as well as prediction error. They do not independently demonstrate
miscalibration or unfairness.

Fixed probability-band calibration tables and final-test segment
diagnostics are available in docs/evaluation/ and in the application.
Small probability bands provide unstable estimates; for example, the
final churn band above 50% contained only three episodes.

For nonresponse, later administrative fields used in retrospective
segment analysis are not forecasting inputs.

No confidence intervals or formal significance tests were calculated.
Observed differences are sample estimates.

## 8. Revised Explanations and What-if Functionality

The application now includes:

- Original, revised, and constant-baseline performance comparisons.
- Plain-language performance interpretation and limitations.
- Calibration and segment-error diagnostics.
- Overall permutation importance, clearly separated from explanations
  of one prediction.
- Exact individual logistic contributions relative to mean training
  features.
- Predictor values in individual explanation tables, chart labels,
  tooltips, and written explanations.
- Input roles identifying fixed history, operational conditions,
  external conditions, and derived values.
- Fixed historical coverage gaps and documentation issues.
- Automatic recalculation of administrative complexity.
- Observed numeric ranges and permitted categorical values.
- Rejection of returned notices without a sent notice.
- Rejection of responses without an outreach attempt.
- Clearing of previous scenario results after invalid submissions.
- Side-by-side baseline and scenario inputs, predictions, and changes.
- Automatically generated summaries covering inputs, changes,
  implications, trade-offs, cautions, and limitations.

Contributions are expressed in log odds, not percentage-point effects.
Positive and negative contributions indicate higher or lower model risk
relative to the training reference.

The training reference used for explanations differs from the selected
episode used as the what-if baseline.

Observed response, returned mail, and verification requirements are not
freely controlled actions. Scenario edits represent alternative
conditions at the same snapshot, not proven intervention effects.

## 9. Structured User Test and Resulting Improvement

One student outside the project group, identified as P1, completed a
guided test on November 2, 2026. P1 had no Medicaid experience and was
a substitute for the intended supervisor.

The facilitator explained the concept and supplied two scenario
instructions. Tasks covered baseline interpretation, contributing
factors, scenario comparison, possible decisions, and unclear information.

The participant reviewed:

| Scenario | Baseline risk | Updated risk | Difference |
|---|---:|---:|---:|
| Response changed to Yes | 16.26% | 11.17% | -5.09 percentage points |
| Response Yes and returned notice Yes | 16.26% | 17.53% | +1.26 percentage points |

The participant's reported explanation identified the direction of both
changes, suggested closer case review or contact-information checks,
and recognized that predictions are not guarantees.

P1 suggested displaying actual predictor values beside contributions.
We implemented that change for baseline and scenario explanations in
both models.

The updated application passed automated checks after the change.

This was one assisted session. Exposure to example answers was not
recorded, independent factor identification was not separately
established, and no participant retest of the revised display was
documented. The findings do not establish usability for Medicaid staff.

The complete record is in docs/USER_TESTING.md.

## 10. Verification and Reproducibility

The macOS Python 3.14.6 environment installed the recorded requirements
and passed python -m pip check.

The evaluation scripts verified household separation. The updated
application checks passed after the explanation-display revision.

Application checks covered:

- Consistent predictions for unchanged inputs.
- Reconstruction of individual prediction log odds.
- Fixed historical inputs.
- Recalculated administrative complexity.
- Numeric and categorical validation.
- Loading of all three application pages.
- Modified scenarios with unchanged baselines.
- Rejection of invalid notice and response combinations.
- Clearing stale results and recovery after correcting inputs.

Relevant commands:

    python test_app.py
    python -m pip check
    python -m streamlit run main.py

Evaluation outputs were produced using evaluate_models.py and
evaluate_final.py. Model choices must not be repeatedly changed in
response to final-test results.

## 11. Remaining Limitations and Future Priorities

- All records and outcomes are synthetic.
- The churn model closely matches assumptions used to generate labels;
  performance does not demonstrate validity on real administrative data.
- The original test set was reused from Week 10.
- Evaluation was neither temporal nor external.
- Predictor event timestamps and historical snapshots are incomplete.
- Nonresponse combines uncontacted cases and contacted nonresponders.
- Contact Stability is assumed known at the earlier forecast origin.
- Model revisions produced only tiny probability-error changes.
- Ranking remains modest, and the nonresponse model has many ties.
- Protected attributes are excluded as predictors, but this does not
  establish model fairness.
- Scenario checks cannot establish real-world plausibility.
- Staffing costs, contact burden, financial savings, and intervention
  effectiveness are not estimated.
- User testing involved one guided substitute participant.
- Uncertainty estimates and independent usability retesting remain future
  work.

Priorities are authorized timestamped data, prospective or temporal
evaluation, calibration assessment, subgroup error analysis, operational
capacity and cost information, and testing with renewal operations staff.

## 12. Traceability

The Week 10 submission is preserved by the Git tag:

week10-submission

Week 11 evaluation scripts, saved outputs, application changes, and
testing documentation are recorded in subsequent commits.