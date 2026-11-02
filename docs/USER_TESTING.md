# User Testing — MedChurn Insights

## Participant and Method

- Participant: Tarun (P1), a classmate outside the project group.
- Background: Student with no Medicaid experience.
- Date: November 2, 2026.
- Method: Guided application walkthrough using synthetic data only.

Poornima explained the application concept and supplied scenario
instructions. Tarun was a substitute for the intended Renewal
Operations Supervisor.

## Tasks

Tarun was asked to interpret a baseline prediction, review contributing
factors, compare two scenarios, describe a possible planning use,
and identify confusing or missing information.

## Scenario Comparisons

Model: Coverage churn. Baseline episode: R0000014.

| Scenario | Changes from baseline | Baseline risk | Updated risk | Difference |
|---|---|---:|---:|---:|
| 1 | Response before prediction: No to Yes | 16.26% | 11.17% | -5.09 percentage points |
| 2 | Response: No to Yes; returned notice: No to Yes | 16.26% | 17.53% | +1.26 percentage points |

Both scenarios retained a sent notice and one outreach attempt.
Differences use unrounded predictions.

## Feedback and Observations

Tarun explained that the first scenario lowered estimated risk,
while the second raised it above the original baseline. He suggested
using the results to identify cases needing closer review or
contact-information checks and recognized that predictions are
not guarantees.

He requested actual predictor values beside individual contributions
to make the explanations clearer.

Initial feedback was only "good," so the facilitator requested a
more detailed interpretation. No participant-specific application
error was reported.

## Resulting Application Improvement

Updated pages/3_Predictive_Scenarios.py to display input values in:

- Individual contribution tables.
- Chart labels and tooltips.
- Written explanations.

Binary values now appear as Yes or No. The change applies to baseline
and scenario explanations for both models.

After the update, python test_app.py passed all application checks.
Live screenshots also confirmed the revised display and working
scenarios.

## Limitations

This was one guided test with a student unfamiliar with Medicaid.
Scenario changes were supplied, and independent scenario creation
and factor identification were not fully documented. Exposure to
AI-generated practice answers was not recorded.

The test provides preliminary usability feedback, not evidence of
operational effectiveness. Predictions and scenario differences
are model estimates, not proven causal effects or guarantees.