# User Testing — MedChurn Insights

## Participant and Method

- Participant: Tarun (P1), a classmate outside the project group.
- Background: Student with no Medicaid experience.
- Test date: November 2, 2026.
- Method: Guided walkthrough using synthetic data only.

Poornima explained the application concept and provided scenario
instructions. Tarun served as a substitute for the intended Renewal
Operations Supervisor.

## Tasks

The walkthrough covered:

1. Interpreting a baseline prediction.
2. Reviewing individual predictor contributions.
3. Running and comparing two what-if scenarios.
4. Identifying a possible planning decision.
5. Identifying confusing or missing information.

## Scenario Results

Model: Coverage churn.
Baseline episode: R0000014.

| Scenario | Changes from baseline | Baseline risk | Updated risk | Difference |
|---|---|---:|---:|---:|
| 1 | Responded Before Prediction: No to Yes | 16.26% | 11.17% | -5.09 percentage points |
| 2 | Responded Before Prediction: No to Yes; Renewal Notice Returned: No to Yes | 16.26% | 17.53% | +1.26 percentage points |

Both scenarios retained a sent notice and one outreach attempt.
Both passed input validation. Differences use unrounded predictions.

The scenarios represent alternative case conditions, not proven
effects of interventions.

## Feedback and Observations

Tarun initially described both results as "good." After a follow-up
question, his reported explanation was:

> In the first situation, the estimated risk drops from about 16%
> to 11%. In the second situation, the risk goes up to 17.5%,
> making it higher than where it started. A supervisor can look
> at these changes to decide which cases need a second look or
> a double-check on their contact information. Keep in mind,
> these numbers are just educated guesses—not guarantees.

This response identified the direction of the comparisons,
a possible supportive-review use, and uncertainty about outcomes.

Tarun suggested showing actual predictor values beside their
contributions to make the explanations clearer.

No participant-specific application error was reported.
The facilitator provided conceptual guidance and scenario instructions.

## Improvement Based on Feedback

Updated pages/3_Predictive_Scenarios.py to display predictor values in:

- Individual contribution tables.
- Chart labels and tooltips.
- Written contribution explanations.

Binary values appear as Yes or No. For example:
"Responded Before Prediction: Yes."

The improvement applies to baseline and scenario explanations
for both models without changing model predictions.

## Verification

After the update, python test_app.py passed all application checks,
including prediction consistency, explanation reconstruction,
input constraints, fixed baselines, invalid-input handling,
and recovery after correction.

Live application screenshots confirmed the Input Value column
and working scenarios for both models.

These checks verify application behavior; they do not establish
improved participant comprehension.

## Limitations

This was one guided session with a student unfamiliar with Medicaid
operations. The facilitator supplied the scenario changes.

Independent identification of the main contributing factors,
participant-created scenarios, and reasoning about scenario
plausibility were not fully documented. Exposure to AI-generated
practice answers was not recorded.

The findings provide preliminary usability feedback and do not
establish usefulness for Medicaid staff or real-world predictive
validity. Further testing should address the undocumented tasks
and include renewal operations staff.