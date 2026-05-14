"""
Custom library screening script using the QSAR pipeline infrastructure
but training and screening using only ECFP4 and MACCS fingerprints.

This script addresses an issue encountered with the original
`library_screen.py`, where many molecules in external libraries were
discarded due to missing or undefined physicochemical descriptors when
screening.  To maximise the number of candidate molecules retained, we
omit physicochemical descriptors from both the training features and
the screening workflow.  Instead, we rely solely on extended
connectivity fingerprints (ECFP4, radius 2) and MACCS keys.  A
similarity‐based applicability domain is still estimated from the
training set using ECFP bit vectors.

The script performs the following steps:

1. Loads and cleans a ChEMBL activity dataset (e.g. BTK inhibitors).
2. Converts IC₅₀ values to pIC₅₀.
3. Computes ECFP and MACCS features for each molecule.
4. Splits the data into training and test sets based on Bemis–Murcko
   scaffolds.
5. Trains an XGBoost regressor on the training features.
6. Estimates an applicability domain (AD) threshold based on the
   training set, using the 5th percentile of mean similarities over k
   nearest neighbours.
7. For each provided library file, standardises SMILES strings,
   computes ECFP/MACCS features, predicts pIC₅₀ values, computes
   mean similarities to the training set, flags out‐of‐domain compounds,
   calculates simple drug‐likeness properties, and ranks compounds by
   predicted pIC₅₀.

The results for each library are saved as CSV files in the specified
output directory.  Each output includes the canonical SMILES, predicted
pIC₅₀, mean similarity, AD flag, and basic properties.  Optionally,
drug‐likeness filtering can be applied.

Example usage:

```
python library_screen_custom.py \
    --target_csv btk.csv \
    --library_csv reference_ligands_from_sdf.csv l1300_fda_approved_library_input.csv l1400_natural_product_library_input.csv \
    --output_dir custom_results \
    --top_n_hits 100
```

"""

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
# Import rdMolStandardize from the correct submodule.  Note: direct import
# from rdkit.Chem.rdMolStandardize may fail in certain versions, so use
# rdkit.Chem.MolStandardize instead (mirroring pipeline.py).
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import rdMolDescriptors
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity

from qsar_pipeline.pipeline import (
    load_chembl_dataset,
    clean_and_standardise_smiles,
    convert_ic50_to_pic50,
    compute_features,
    _assemble_feature_matrix,
    scaffold_split,
    estimate_applicability_domain,
    _compute_ecfp4_fingerprint,
    _compute_maccs_fingerprint,
)
from xgboost import XGBRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a QSAR model (ECFP+MACCS only) and screen external libraries."
    )
    parser.add_argument(
        "--target_csv",
        type=str,
        required=True,
        help="Path to the ChEMBL activity CSV (e.g. btk.csv).",
    )
    parser.add_argument(
        "--library_csv",
        type=str,
        nargs="+",
        required=True,
        help="Paths to screening library CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write screening results.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of target data used for testing (for reporting only).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top_n_hits",
        type=int,
        default=100,
        help="Number of top hits to save for each library.",
    )
    parser.add_argument(
        "--druglike",
        action="store_true",
        default=False,
        help="Apply simple drug‑likeness filtering (Lipinski‑like).",
    )
    return parser.parse_args()


def train_model(target_csv: str, test_fraction: float, random_state: int) -> tuple[
    XGBRegressor, List[object], float
]:
    """Train an XGBoost regressor using ECFP+MACCS features only.

    Parameters
    ----------
    target_csv : str
        Path to the ChEMBL activity CSV containing at least columns ``Smiles``,
        ``Standard Relation``, and ``Standard Value``.
    test_fraction : float
        Fraction of the curated data reserved for the test set (for reporting only).
    random_state : int
        Random seed used for reproducibility.

    Returns
    -------
    model : XGBRegressor
        Fitted model trained on the training portion of the data.
    train_fps : list of rdkit.DataStructs.cDataStructs.ExplicitBitVect
        List of fingerprint bit vectors for training molecules (used for AD).
    ad_threshold : float
        Applicability domain threshold computed from the training set.
    """
    # Load and curate data
    df_raw = load_chembl_dataset(target_csv)
    df_clean = clean_and_standardise_smiles(df_raw, smiles_column="Smiles")
    df_pic50 = convert_ic50_to_pic50(df_clean, value_column="Standard Value")
    if len(df_pic50) < 10:
        raise RuntimeError("Too few data points remain after curation – aborting.")
    # Compute features; returns ecfp, maccs, descriptors, descriptor_names, ecfp_bv_list
    ecfp, maccs, desc, descriptor_names, ecfp_bv_list = compute_features(df_pic50)
    # Align DataFrame
    df_pic50 = df_pic50.iloc[: len(ecfp)].reset_index(drop=True)
    # Assemble feature matrix using only fingerprints (ECFP) and MACCS
    X = _assemble_feature_matrix(ecfp, maccs, desc, use_fingerprints=True, use_maccs=True, use_desc=False)
    y = df_pic50["pIC50"].values
    # Train‑test split (scaffold based)
    train_idx, test_idx = scaffold_split(df_pic50, test_fraction=test_fraction, random_state=random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Keep training fingerprint bit vectors for AD
    train_fps = [ecfp_bv_list[i] for i in train_idx]
    # Train XGBRegressor
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method="hist",
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    # Evaluate (for reporting)
    if len(y_test) > 1:
        preds = model.predict(X_test)
        r2 = float(np.corrcoef(y_test, preds)[0, 1] ** 2)
        rmse = float(np.sqrt(((y_test - preds) ** 2).mean()))
        mae = float(np.abs(y_test - preds).mean())
        print(f"Test metrics (ECFP+MACCS): R² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
    # Applicability domain threshold
    _, ad_threshold = estimate_applicability_domain(train_fps, train_fps, k=5, percentile=5.0)
    print(f"Applicability domain threshold = {ad_threshold:.3f}")
    return model, train_fps, ad_threshold


def screen_library_custom(
    smiles_list: List[str],
    model: XGBRegressor,
    train_fps: List[object],
    ad_threshold: float,
    k: int = 5,
    druglike: bool = False,
    top_n_hits: int | None = None,
) -> pd.DataFrame:
    """Screen a list of SMILES using ECFP+MACCS features and the trained model.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings to screen.
    model : XGBRegressor
        Fitted regression model.
    train_fps : list of ExplicitBitVect
        Fingerprint bit vectors of the training set for AD computation.
    ad_threshold : float
        Applicability domain threshold computed from the training set.
    k : int, default 5
        Number of nearest neighbours used in similarity calculations.
    druglike : bool, default False
        Whether to apply drug‑likeness filters (Lipinski‑like rules).
    top_n_hits : int or None, default None
        If provided, truncate the returned DataFrame to the top N compounds
        ranked by predicted pIC50.

    Returns
    -------
    screen_df : pandas.DataFrame
        DataFrame containing canonical SMILES, predicted pIC50, mean similarity,
        AD flag, and basic molecular properties.
    """
    records: List[dict] = []
    chooser = rdMolStandardize.LargestFragmentChooser()
    # Preload training fingerprint list for BulkTanimotoSimilarity
    train_fp_list = list(train_fps)
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = chooser.choose(mol)
            Chem.SanitizeMol(mol)
            # Skip molecules without carbon atoms
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]")):
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            # Compute ECFP and MACCS arrays and bit vector
            ecfp_arr = _compute_ecfp4_fingerprint(mol)
            maccs_arr = _compute_maccs_fingerprint(mol)
            fp_bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            # Construct feature vector
            X = np.concatenate([ecfp_arr, maccs_arr])
            # Predict pIC50
            pred = float(model.predict(X.reshape(1, -1))[0])
            # Compute mean similarity to training set
            sims = BulkTanimotoSimilarity(fp_bv, train_fp_list)
            sims = np.sort(np.array(sims))[-k:]
            mean_sim = sims.mean()
            in_ad = mean_sim >= ad_threshold
            # Basic properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot = Descriptors.NumRotatableBonds(mol)
            # Drug‑likeness filter
            if druglike:
                if (
                    mw > 500
                    or logp > 5
                    or hbd > 5
                    or hba > 10
                    or rot > 10
                ):
                    continue
            records.append(
                {
                    "smiles": can_smi,
                    "prediction": pred,
                    "mean_similarity": mean_sim,
                    "in_AD": in_ad,
                    "MolWt": mw,
                    "MolLogP": logp,
                    "HBD": hbd,
                    "HBA": hba,
                    "RotatableBonds": rot,
                }
            )
        except Exception:
            continue
    screen_df = pd.DataFrame(records)
    if screen_df.empty:
        return screen_df
    screen_df = screen_df.sort_values(by="prediction", ascending=False).reset_index(drop=True)
    if top_n_hits is not None and top_n_hits > 0:
        screen_df = screen_df.head(top_n_hits)
    return screen_df


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Train model and compute AD threshold
    model, train_fps, ad_threshold = train_model(
        target_csv=args.target_csv,
        test_fraction=args.test_fraction,
        random_state=args.random_state,
    )
    # Screen each library
    for lib_path in args.library_csv:
        base = os.path.splitext(os.path.basename(lib_path))[0]
        print(f"Screening {base} using ECFP+MACCS model...")
        try:
            lib_df = pd.read_csv(lib_path)
        except Exception as exc:
            print(f"Warning: could not read {lib_path}: {exc}")
            continue
        # Detect SMILES column (case‑insensitive)
        smiles_col = None
        for col in lib_df.columns:
            if col.lower() == "smiles":
                smiles_col = col
                break
        if smiles_col is None:
            print(f"Warning: no SMILES column found in {lib_path}; skipping.")
            continue
        smiles_list = lib_df[smiles_col].dropna().astype(str).tolist()
        screen_df = screen_library_custom(
            smiles_list=smiles_list,
            model=model,
            train_fps=train_fps,
            ad_threshold=ad_threshold,
            k=5,
            druglike=args.druglike,
            top_n_hits=args.top_n_hits,
        )
        out_file = os.path.join(args.output_dir, f"{base}_custom_screen_results.csv")
        screen_df.to_csv(out_file, index=False)
        print(f"Saved {len(screen_df)} hits for {base} to {out_file}")


if __name__ == "__main__":
    main()