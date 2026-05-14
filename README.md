# Deep Learning Virtual Screening

AI-driven virtual screening and QSAR pipeline using machine learning and deep learning for drug discovery.

## Project aim

This repository is designed as a publication-quality starting framework for ligand-based virtual screening. The workflow begins with compound activity data, performs careful data cleaning, generates molecular fingerprints/descriptors, trains multiple regression models, evaluates model quality, and applies the final model to screen new compounds.

## Planned workflow

1. Collect bioactivity data from ChEMBL or curated CSV files.
2. Standardize molecules and remove invalid SMILES.
3. Apply drug-likeness and basic ADMET-style filters.
4. Generate ECFP4 fingerprints and molecular descriptors.
5. Train baseline and ensemble regression models.
6. Evaluate models using cross-validation and external test data.
7. Generate publication-quality plots.
8. Screen external compound libraries.
9. Prioritize hits for docking and molecular dynamics in a later stage.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── screening_library/
├── notebooks/
├── results/
│   ├── figures/
│   ├── models/
│   └── tables/
├── src/
│   ├── data_preparation.py
│   ├── featurization.py
│   ├── train_ml_models.py
│   ├── evaluate_models.py
│   └── screen_library.py
├── requirements.txt
└── README.md
```

## First target pipeline

The first implementation will focus on a single-target QSAR regression model using:

- SMILES and activity values from ChEMBL or a curated CSV file
- pIC50 as the target variable
- ECFP4 fingerprints
- Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, and CatBoost models
- Cross-validation and external test-set evaluation
- Screening of new compounds using the best saved model

## Notes

Docking and molecular dynamics will be kept as downstream validation steps and will not be part of the first ML/DL model-building stage.
