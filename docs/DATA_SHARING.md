# Data Provenance and Sharing

## Data Origin

All beneficiary, household, renewal, outreach, administrative,
and churn records were generated synthetically using the
included Python scripts and fixed random seeds.

No actual Medicaid administrative records or real beneficiary
identifiers were used. The data were not calibrated or validated
against real Medicaid records.

## Included Data

The project includes:

- data/raw/: synthetic source tables.
- data/final/medicaid_modeling_dataset.csv: inputs and outcomes
  used by the predictive models.
- data/final/medicaid_fairness_audit.csv: synthetic attributes
  and outcomes used for descriptive subgroup analysis.
- code/: data-generation and validation scripts.

The application runs from the included final CSV files without
requiring access to an external dataset.

## Sharing and Access

No restricted external dataset is required for this project.
Contributors must confirm authorization to share their project
materials before publishing them publicly.

The instructor and all contributors must have access to the
submitted repository or folder.

If restricted real data are introduced in the future, those files
must be excluded from public distribution. Documentation must
identify the data custodian, restrictions, approved access
procedure, and where authorized users should place the files
to run the application.

## Interpretation

These data support educational demonstrations only.

Synthetic identifiers do not refer to real beneficiaries.
Model performance and scenario results must not be presented
as findings validated on a real Medicaid population.