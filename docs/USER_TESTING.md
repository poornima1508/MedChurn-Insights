# User Testing — MedChurn Insights

## Purpose

Assess whether a person outside the project group can interpret a
prediction, review contributing factors, compare scenarios, and
identify a possible planning use.

## Participant and Method

- Participant: P1, a student outside the project group.
- Medicaid experience: None.
- Test date: November 2, 2026.
- Data: Synthetic renewal records only.
- Method: Guided walkthrough of the local Streamlit application.

The facilitator explained the application's underlying concept
and supplied instructions for two scenarios. Whether the participant
viewed the AI-generated example answers before responding was not
recorded. Findings therefore describe assisted use, rather than
independent task completion.

P1 was a substitute for the intended Renewal Operations Supervisor.
This test does not establish usefulness for experienced Medicaid staff.

## Tasks

The participant was asked to:

1. Interpret the baseline coverage-churn prediction.
2. Identify factors raising or lowering that prediction.
3. Run and compare two plausible scenarios with the same baseline.
4. Describe a decision the results could inform.
5. Identify confusing or missing information.

## Scenario Results

Model: Coverage churn.
Baseline episode: R0000014.

| Scenario | Changed inputs from baseline | Baseline risk | Scenario risk | Difference |
|---|---|---:|---:|---:|
| 1 | Responded Before Prediction: 0 to 1 | 16.26% | 11.17% | -5.09 percentage points |
| 2 | Responded Before Prediction: 0 to 1; Renewal Notice Returned: 0 to 1 | 16.26% | 17.53% | +1.26 percentage points |

Both scenarios retained one outreach attempt and a sent renewal
notice. They passed the application's input checks. Passing these
checks does not establish real-world plausibility.

Differences were calculated using unrounded predictions. Displayed
percentages may therefore differ slightly when subtracted.

These scenarios compare alternative case conditions at the same
prediction snapshot. They do not estimate the causal effects of
outreach or other interventions.

## Feedback and Interpretation

The participant initially described both scenario results as "good."

The facilitator subsequently reported this participant explanation:

> In the first situation, the estimated risk drops from about 16%
> to 11%. In the second situation, the risk goes up to 17.5%,
> making it higher than where it started. A supervisor can look
> at these changes to decide which cases need a second look or
> a double-check on their contact information. Keep in mind,
> these numbers are just educated guesses—not guarantees.

This response identifies the direction of both comparisons,
a possible supportive-review use, and uncertainty about outcomes.

Independent baseline interpretation and identification of individual
contributing factors were not separately established in the recorded
notes. Earlier example wording was AI-assisted and is not treated
as evidence of unaided participant understanding.

## Identified Usability Issue

P1 suggested displaying each predictor's actual value beside its
individual contribution.

A predictor name alone can be ambiguous. For example,
"Responded Before Prediction" does not show whether the explanation
applies to a response of Yes or No.

## Application Improvement

Updated pages/3_Predictive_Scenarios.py to:

- Add an Input Value column to individual contribution tables.
- Include predictor values in chart labels and tooltips.
- Include predictor values in written contribution explanations.
- Display binary values as Yes or No.
- Apply the improvement to baseline and scenario explanations
  for both models.

Example: "Responded Before Prediction: Yes."

This addresses the reported ambiguity without changing the models.

## Verification

After the update, python test_app.py completed successfully.

Checks covered prediction consistency, explanation reconstruction,
fixed history, derived complexity, input validation, page loading,
unchanged and modified scenarios, fixed baselines, rejection of
invalid combinations, clearing stale results, and recovery after
correcting inputs.

These automated checks verify application behavior. They do not
establish that the revised explanations are easier for users to
understand. A participant retest of the revised display has not
been documented.

## Limitations and Next Steps

- Only one outside participant was involved.
- The participant had no Medicaid operational experience.
- Tasks were guided, and scenario changes were supplied.
- Exposure to example answers was not recorded.
- No formal timing or independent task-success scores were collected.
- Findings do not establish operational effectiveness or model accuracy.
- Future testing should include renewal operations staff and assess
  whether the revised display improves independent interpretation.