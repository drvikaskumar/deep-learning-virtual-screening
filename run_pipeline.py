"""
Example script to run the QSAR pipeline on a single ChEMBL target dataset.

Usage:
    python run_pipeline.py --input_csv /path/to/btk.csv --output_dir ./results

The script performs the following steps:

1. Load and filter the input dataset to retain only IC50 values expressed in nM.
2. Clean and standardise the SMILES strings, removing salts, inorganics and
   duplicates.
3. Convert IC50 values to pIC50.
4. Compute ECFP4 fingerprints, MACCS keys and physicochemical descriptors.
5. Perform a scaffold‑based split into training and test sets.
6. Train multiple regression models using 5‑fold cross‑validation and select
   the best performing model.
7. Evaluate the selected model on the test set and estimate an applicability
   domain threshold based on the training fingerprints.
8. Generate SHAP feature importance plots for the top model.
9. (Optional) Screen an external library of SMILES using the trained model.

Outputs are saved in the specified `--output_dir` directory:

* `cross_validation_results.csv` – summary of model performance during cross‑validation.
* `test_predictions.csv` – predictions for the held‑out test set.
* `screening_results.csv` – ranked list of screened compounds (if a library was provided).
* `shap_summary.png` – bar plot of SHAP mean absolute feature importances.

This script is intended for demonstration and can be modified to suit other
targets or modelling strategies.
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from pipeline import (
    load_chembl_dataset,
    clean_and_standardise_smiles,
    convert_ic50_to_pic50,
    compute_features,
    scaffold_split,
    _assemble_feature_matrix,
    train_and_evaluate_models,
    estimate_applicability_domain,
    compute_shap_importances,
    screen_library,
)


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(description='Run QSAR regression pipeline.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the ChEMBL CSV file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to write results.')
    parser.add_argument('--screen_csv', type=str, default=None, help='Optional CSV file with a column named "Smiles" to screen.')
    parser.add_argument('--test_fraction', type=float, default=0.2, help='Fraction of data used for the test set.')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross‑validation.')
    parser.add_argument('--top_n_hits', type=int, default=100, help='Number of top screening hits to save.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Step 1: load dataset
    print('Loading dataset...')
    df_raw = load_chembl_dataset(args.input_csv)
    print(f'Loaded {len(df_raw)} activity records.')
    # Step 2: clean SMILES
    print('Standardising SMILES and removing salts/inorganics...')
    df_clean = clean_and_standardise_smiles(df_raw, smiles_column='Smiles')
    print(f'After curation: {len(df_clean)} unique molecules.')
    # Step 3: convert IC50 to pIC50
    df_pic50 = convert_ic50_to_pic50(df_clean, value_column='Standard Value')
    print(f'Retained {len(df_pic50)} molecules with numeric IC50 values.')
    if len(df_pic50) < 10:
        print('Error: too few molecules remain after curation for model training.')
        return
    # Step 4: compute features
    print('Computing molecular features...')
    ecfp, maccs, desc, descriptor_names, ecfp_bv_list = compute_features(df_pic50)
    # Align df_pic50 with computed features
    df_pic50 = df_pic50.iloc[:len(ecfp)].reset_index(drop=True)
    # Assemble combined feature matrix
    X = _assemble_feature_matrix(ecfp, maccs, desc, use_fingerprints=True, use_maccs=True, use_desc=True)
    y = df_pic50['pIC50'].values
    # Step 5: scaffold split
    print('Performing scaffold‑based train/test split...')
    train_idx, test_idx = scaffold_split(df_pic50, test_fraction=args.test_fraction, random_state=args.random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ecfp_train, ecfp_test = ecfp[train_idx], ecfp[test_idx]
    ecfp_bv_train = [ecfp_bv_list[i] for i in train_idx]
    ecfp_bv_test = [ecfp_bv_list[i] for i in test_idx]
    # Step 6: train models and perform cross‑validation
    print('Training models and evaluating via cross‑validation...')
    results = train_and_evaluate_models(X_train, y_train, n_splits=args.n_splits, random_state=args.random_state)
    # Save cross‑validation results
    cv_df = pd.DataFrame({
        'model': [r.name for r in results],
        'cv_R2': [r.cv_r2 for r in results],
        'cv_RMSE': [r.cv_rmse for r in results],
        'cv_MAE': [r.cv_mae for r in results],
    })
    cv_df.to_csv(os.path.join(args.output_dir, 'cross_validation_results.csv'), index=False)
    print('Cross‑validation results saved.')
    # Select top model
    best_result = results[0]
    best_model = best_result.model
    print(f'Selected best model: {best_result.name} (CV R² = {best_result.cv_r2:.3f})')
    # Retrain best model on full training set
    print('Fitting best model on training set...')
    best_model.fit(X_train, y_train)
    # Step 7: evaluate on test set
    test_preds = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f'Test R² = {test_r2:.3f}, RMSE = {test_rmse:.3f}, MAE = {test_mae:.3f}')
    # Save test predictions
    test_df = df_pic50.iloc[test_idx].copy()
    test_df['predicted_pIC50'] = test_preds
    test_df['residual'] = test_df['pIC50'] - test_df['predicted_pIC50']
    test_df[['clean_smiles', 'pIC50', 'predicted_pIC50', 'residual']].to_csv(
        os.path.join(args.output_dir, 'test_predictions.csv'), index=False
    )
    print('Test predictions saved.')
    # Step 8: applicability domain estimation
    print('Estimating applicability domain threshold...')
    # Compute threshold using only training data
    # Estimate applicability domain using bit vectors
    in_domain_mask_train, ad_threshold = estimate_applicability_domain(np.array(ecfp_bv_train, dtype=object), np.array(ecfp_bv_train, dtype=object), k=5, percentile=5.0)
    print(f'Applicability domain threshold (mean similarity) = {ad_threshold:.3f}')
    # Step 9: SHAP analysis
    print('Computing SHAP feature importances...')
    # Build feature names: indices for ECFP4, MACCS and descriptors
    fp_names = [f'ECFP4_{i}' for i in range(ecfp.shape[1])] + [f'MACCS_{i}' for i in range(maccs.shape[1])] + descriptor_names
    shap_fig = compute_shap_importances(best_model, X_train, feature_names=fp_names, n_samples=300, output_dir=args.output_dir)
    if shap_fig:
        print(f'SHAP summary plot saved to {shap_fig}')
    else:
        print('SHAP computation skipped or failed.')
    # Step 10: optional screening
    if args.screen_csv:
        print('Screening external library...')
        try:
            lib_df = pd.read_csv(args.screen_csv)
        except Exception as e:
            print(f'Error loading screening file: {e}')
            lib_df = None
        if lib_df is not None:
            if 'Smiles' not in lib_df.columns:
                print('Error: screening CSV must contain a column named "Smiles".')
            else:
                smiles_list = lib_df['Smiles'].tolist()
                screen_df = screen_library(smiles_list, best_model, ecfp_bv_train, ad_threshold, k=5, druglike=True)
                # Keep top hits
                screen_df = screen_df.head(args.top_n_hits)
                screen_df.to_csv(os.path.join(args.output_dir, 'screening_results.csv'), index=False)
                print(f'Screening results saved to screening_results.csv ({len(screen_df)} hits).')
    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()
