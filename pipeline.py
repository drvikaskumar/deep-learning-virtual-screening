"""
QSAR pipeline implementation for single‑target regression modelling.

This module defines a set of functions that implement the end‑to‑end workflow for
building, evaluating and applying QSAR models to a ChEMBL‑derived dataset.  It
follows best practices for chemical data curation【962878708436635†L160-L167】,
molecular representation【851742792911846†L1090-L1104】 and model validation【851742792911846†L1189-L1203】.  The
main entry points are:

* `load_chembl_dataset` – load and filter a ChEMBL CSV file containing activity
  data (IC50 values in nanomolar units).
* `clean_and_standardise_smiles` – standardise SMILES strings, remove salts and
  inorganics, sanitise molecules and generate canonical SMILES.
* `convert_ic50_to_pic50` – convert IC50 values in nM to pIC50【171224809600439†L189-L197】.
* `compute_features` – calculate ECFP4 fingerprints, MACCS keys and a set of
  physicochemical descriptors for each molecule.
* `scaffold_split` – split the dataset into training and test sets based on
  Bemis–Murcko scaffolds.
* `train_and_evaluate_models` – train several regression algorithms using
  cross‑validation and return the fitted models and performance metrics.
* `estimate_applicability_domain` – compute a similarity‑based applicability
  domain using k‑nearest neighbours on fingerprint space【851742792911846†L1189-L1203】.
* `compute_shap_importances` – calculate SHAP values for the top model to
  identify important features【851742792911846†L1215-L1220】.
* `screen_library` – apply the trained model to an external library of SMILES,
  filter compounds according to the applicability domain and simple
  drug‑likeness rules, and return a ranked list of predicted pIC50 values.

Example usage can be found in `run_pipeline.py`.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, MACCSkeys
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import shap
import matplotlib.pyplot as plt


def load_chembl_dataset(csv_path: str) -> pd.DataFrame:
    """Load a ChEMBL dataset from a CSV file and retain only IC50 data in nM.

    The CSV file is expected to contain at least the following columns:

      * `Smiles` – canonical SMILES string.
      * `Standard Type` – type of activity measurement (e.g. 'IC50').
      * `Standard Relation` – relation symbol (e.g. '=').
      * `Standard Value` – numeric IC50 values.
      * `Standard Units` – units of the activity values (expected 'nM').

    Rows that do not satisfy these criteria are dropped.  A copy of the
    filtered DataFrame is returned.
    """
    df = pd.read_csv(csv_path, sep=';')
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    # Apply filters: keep IC50 records with nanomolar units and equality relations
    type_mask = df['Standard Type'].str.lower().str.contains('ic50', na=False)
    units_mask = df['Standard Units'].str.lower().str.contains('nm', na=False)
    # Accept relations that contain '=' (e.g. '=', "'=" etc.)
    relation_mask = df['Standard Relation'].astype(str).str.contains('=', na=False)
    filtered = df[type_mask & units_mask & relation_mask].copy()
    return filtered


def clean_and_standardise_smiles(df: pd.DataFrame, smiles_column: str = 'Smiles') -> pd.DataFrame:
    """Standardise SMILES strings and remove problematic records.

    For each SMILES string in the input DataFrame:

      * Parse the SMILES into an RDKit molecule.
      * Keep only the largest fragment to remove salts/mixtures.
      * Sanitize the molecule to ensure valence and aromaticity are consistent.
      * Require at least one carbon atom (i.e. organic molecules).
      * Generate a canonical SMILES with stereochemistry.

    Molecules that cannot be parsed or sanitised are skipped.  Duplicates are
    removed based on the canonical SMILES.  The resulting DataFrame contains
    additional columns `clean_smiles` and `rdkit_mol`.

    This procedure implements standard curation steps described by Fourches
    et al. 【962878708436635†L160-L167】, namely the removal of salts, inorganic compounds and duplicates.
    """
    canonical_smiles: List[str] = []
    rdkit_mols: List[Chem.Mol] = []
    indices: List[int] = []
    chooser = rdMolStandardize.LargestFragmentChooser()
    for idx, smi in enumerate(df[smiles_column]):
        if pd.isna(smi):
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            # choose largest fragment to remove salts/solvents
            mol = chooser.choose(mol)
            # sanitise (may raise a ValueError)
            Chem.SanitizeMol(mol)
            # keep only organic molecules containing at least one carbon
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]')):
                continue
            # canonicalise with stereochemistry
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            canonical_smiles.append(can_smi)
            rdkit_mols.append(mol)
            indices.append(idx)
        except Exception:
            continue
    curated = df.iloc[indices].copy().reset_index(drop=True)
    curated['clean_smiles'] = canonical_smiles
    curated['rdkit_mol'] = rdkit_mols
    # remove duplicates
    curated = curated.drop_duplicates(subset='clean_smiles').reset_index(drop=True)
    return curated


def convert_ic50_to_pic50(df: pd.DataFrame, value_column: str = 'Standard Value') -> pd.DataFrame:
    """Convert IC50 values (in nM) to pIC50 and drop rows with missing values.

    The conversion uses the formula pIC50 = 9 – log10(IC50_nM).  This is
    equivalent to taking the negative logarithm of the IC50 value expressed in
    molar concentration; an IC50 of 1 nM corresponds to pIC50 = 9, 10 nM to 8,
    100 nM to 7, etc., as illustrated by Navre【171224809600439†L189-L197】.
    """
    ic50_values = pd.to_numeric(df[value_column], errors='coerce')
    df = df.loc[~ic50_values.isna()].copy()
    df['IC50_nM'] = ic50_values.loc[~ic50_values.isna()].astype(float)
    # Avoid non-positive values
    df = df[df['IC50_nM'] > 0].copy()
    df['pIC50'] = 9.0 - np.log10(df['IC50_nM'])
    return df.reset_index(drop=True)


def _compute_ecfp4_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    """Compute an ECFP4 (radius 2) fingerprint as a numpy array of bits."""
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def _compute_maccs_fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Compute the 166‑bit MACCS keys fingerprint as a numpy array."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((fp.GetNumBits(),), dtype=np.uint8)
    ConvertToNumpyArray(fp, arr)
    return arr


def _compute_rdkit_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute a selected set of physicochemical descriptors for a molecule.

    Descriptors include molecular weight, logP, topological polar surface area,
    number of hydrogen bond donors/acceptors, rotatable bonds, ring count,
    fraction of sp³ carbon atoms, heavy atom count and molar refractivity.  These
    descriptors provide global information complementary to fingerprints.
    """
    # Define descriptor functions and names
    desc_funcs = [
        ("MolWt", Descriptors.MolWt),
        ("MolLogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("NumHDonors", Descriptors.NumHDonors),
        ("NumHAcceptors", Descriptors.NumHAcceptors),
        ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        ("RingCount", Descriptors.RingCount),
        ("FractionCSP3", Descriptors.FractionCSP3),
        ("HeavyAtomCount", Descriptors.HeavyAtomCount),
        ("MolMR", Descriptors.MolMR),
    ]
    values: List[float] = []
    for _, func in desc_funcs:
        try:
            val = func(mol)
        except Exception:
            val = np.nan
        values.append(val)
    return np.array(values, dtype=float)


def compute_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[object]]:
    """Compute fingerprint and descriptor arrays for each molecule in the DataFrame.

    Returns a tuple `(ecfp, maccs, descriptors, descriptor_names)` where:

      * `ecfp` is a 2‑D array of shape (n_samples, 2048) containing ECFP4
        fingerprints.
      * `maccs` is a 2‑D array of shape (n_samples, 166) containing MACCS keys.
      * `descriptors` is a 2‑D array of shape (n_samples, n_descriptors)
        containing physicochemical descriptors.
      * `descriptor_names` is a list of descriptor names corresponding to the
        columns of `descriptors`.

    If any descriptor is NaN for a molecule (e.g. due to unusual structure), the
    corresponding row is removed so that all feature matrices share the same
    length.
    """
    ecfp_list: List[np.ndarray] = []
    maccs_list: List[np.ndarray] = []
    desc_list: List[np.ndarray] = []
    ecfp_bv_list: List[object] = []
    valid_indices: List[int] = []
    # Precompute descriptor names
    descriptor_names = [
        "MolWt",
        "MolLogP",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "RingCount",
        "FractionCSP3",
        "HeavyAtomCount",
        "MolMR",
    ]
    for i, mol in enumerate(df['rdkit_mol']):
        try:
            ecfp = _compute_ecfp4_fingerprint(mol)
            # compute RDKit bit vector for AD
            fp_bv = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            maccs = _compute_maccs_fingerprint(mol)
            descs = _compute_rdkit_descriptors(mol)
            # skip molecules with any NaN descriptor
            if np.isnan(descs).any():
                continue
            ecfp_list.append(ecfp)
            maccs_list.append(maccs)
            desc_list.append(descs)
            ecfp_bv_list.append(fp_bv)
            valid_indices.append(i)
        except Exception:
            continue
    # Reduce DataFrame to valid molecules
    if len(valid_indices) != len(df):
        df = df.iloc[valid_indices].reset_index(drop=True)
    ecfp_arr = np.vstack(ecfp_list)
    maccs_arr = np.vstack(maccs_list)
    desc_arr = np.vstack(desc_list)
    return ecfp_arr, maccs_arr, desc_arr, descriptor_names, ecfp_bv_list


def scaffold_split(df: pd.DataFrame, test_fraction: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a scaffold‑based split of the dataset into train and test indices.

    Bemis–Murcko scaffolds are computed for each molecule and grouped.  Entire
    scaffolds are then assigned to either the training or test set to reduce
    structural overlap between splits.  The fraction of data placed in the test
    set is controlled by `test_fraction`.

    Returns a tuple `(train_idx, test_idx)` containing the integer indices of
    the training and test rows in the DataFrame.
    """
    scaffolds: Dict[str, List[int]] = {}
    for idx, mol in enumerate(df['rdkit_mol']):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = Chem.MolToSmiles(mol)
        scaffolds.setdefault(scaffold, []).append(idx)
    # Sort scaffolds by descending frequency to ensure large scaffolds are split first
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    train_idx: List[int] = []
    test_idx: List[int] = []
    total_count = len(df)
    np.random.seed(random_state)
    for scaffold_indices in scaffold_sets:
        if len(test_idx) / total_count < test_fraction:
            test_idx.extend(scaffold_indices)
        else:
            train_idx.extend(scaffold_indices)
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _assemble_feature_matrix(ecfp: np.ndarray, maccs: np.ndarray, desc: np.ndarray, use_fingerprints: bool = True, use_maccs: bool = True, use_desc: bool = True) -> np.ndarray:
    """Concatenate selected feature blocks into a single matrix."""
    parts = []
    if use_fingerprints:
        parts.append(ecfp)
    if use_maccs:
        parts.append(maccs)
    if use_desc:
        parts.append(desc)
    return np.concatenate(parts, axis=1)


@dataclass
class ModelResult:
    name: str
    model: object
    cv_r2: float
    cv_rmse: float
    cv_mae: float


def train_and_evaluate_models(X: np.ndarray, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> List[ModelResult]:
    """Train several regression models and evaluate them via cross‑validation.

    The following models are trained with default hyperparameters (except for
    disabling verbose output in gradient boosting models):

      * Random Forest Regressor
      * XGBoost Regressor
      * LightGBM Regressor
      * CatBoost Regressor
      * Support Vector Regressor (RBF kernel)
      * Multi‑Layer Perceptron Regressor

    Each model is evaluated using K‑fold cross‑validation with the specified
    number of splits.  The mean R², RMSE and MAE across folds are returned for
    each model.  A list of `ModelResult` instances is returned.
    """
    results: List[ModelResult] = []
    # Define models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=random_state, tree_method='hist', objective='reg:squarederror', n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=random_state),
        'CatBoost': CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, random_seed=random_state, loss_function='RMSE', verbose=False),
        'SVR': SVR(kernel='rbf', C=10.0, gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(256, 128), activation='relu', solver='adam', max_iter=200, random_state=random_state),
    }
    # Use KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for name, model in models.items():
        # Determine scoring functions for cross_val_score
        # We'll compute metrics manually because some models require fit inside cross_val_score
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            # Fit model
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            r2_scores.append(r2_score(y_val, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))
            mae_scores.append(mean_absolute_error(y_val, preds))
        results.append(ModelResult(
            name=name,
            model=model,
            cv_r2=float(np.mean(r2_scores)),
            cv_rmse=float(np.mean(rmse_scores)),
            cv_mae=float(np.mean(mae_scores)),
        ))
    # Sort results by descending R2
    results.sort(key=lambda r: r.cv_r2, reverse=True)
    return results


def estimate_applicability_domain(train_fps: List[object] | np.ndarray, test_fps: List[object] | np.ndarray, k: int = 5, percentile: float = 5.0) -> Tuple[np.ndarray, float]:
    """Estimate the similarity‑based applicability domain and return a mask.

    For each training compound, compute its mean Tanimoto similarity to its
    `k` nearest neighbours within the training set.  The threshold is defined
    as the specified percentile of these mean similarities.  For a set of test
    fingerprints, compute the mean similarity of each fingerprint to the
    training fingerprints (based on k nearest neighbours) and mark those below
    the threshold as out‑of‑domain.  Returns a boolean array indicating whether
    each test compound is inside the domain and the threshold value.

    This procedure implements the similarity‑based applicability domain
    described by Morales‑Ortiz et al.【851742792911846†L1189-L1203】.
    """
    # Compute training similarity distribution
    train_mean_sims: List[float] = []
    for i in range(len(train_fps)):
        sim = BulkTanimotoSimilarity(train_fps[i], list(train_fps))
        # remove self
        sim = np.array(sim)
        sim[i] = 0.0
        top_k = np.sort(sim)[-k:]
        train_mean_sims.append(top_k.mean())
    train_mean_sims = np.array(train_mean_sims)
    threshold = np.percentile(train_mean_sims, percentile)
    # For test fingerprints, compute mean similarity to training fps
    in_domain = []
    for fp in test_fps:
        sims = BulkTanimotoSimilarity(fp, list(train_fps))
        sims = np.sort(np.array(sims))[-k:]
        mean_sim = sims.mean()
        in_domain.append(mean_sim >= threshold)
    return np.array(in_domain, dtype=bool), float(threshold)


def compute_shap_importances(model, X: np.ndarray, feature_names: Optional[List[str]] = None, n_samples: int = 300, output_dir: str = '.') -> str:
    """Compute SHAP values for a tree‑based model and save a bar plot of feature importance.

    Only the first `n_samples` rows of `X` are used to speed up computation.  The
    function attempts to create a SHAP summary bar plot showing the mean absolute
    SHAP value for each feature and saves it to a PNG file in `output_dir`.

    Returns the path to the saved figure.  If SHAP cannot compute explanations
    for the provided model (e.g. non‑tree models), a warning is issued and an
    empty string is returned.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Restrict to the first n_samples to reduce computation time
    background = X[:n_samples]
    try:
        if hasattr(model, 'get_booster') or hasattr(model, 'booster_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, background)
        shap_values = explainer(background)
        # Compute mean absolute SHAP values
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        # Determine feature names
        if feature_names is None:
            feature_names = [f'f{i}' for i in range(X.shape[1])]
        # Create bar plot
        indices = np.argsort(mean_abs)[::-1]
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(mean_abs)), mean_abs[indices])
        plt.xticks(range(len(mean_abs)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Mean |SHAP value|')
        plt.title('Feature importance (SHAP)')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'shap_summary.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        return fig_path
    except Exception as exc:
        warnings.warn(f'Could not compute SHAP values: {exc}')
        return ''


def screen_library(smiles_list: List[str], model, train_fps: np.ndarray, ad_threshold: float, k: int = 5, druglike: bool = True) -> pd.DataFrame:
    """Screen an external list of SMILES strings using the trained model.

    Each SMILES is standardised and featurised.  Predictions are computed
    using the fitted model.  Compounds with mean similarity below `ad_threshold`
    (based on `k` nearest neighbours in the training fingerprint set) are
    flagged as out‑of‑domain.  If `druglike` is True, compounds are filtered
    according to simple drug‑likeness rules (molecular weight ≤ 500, logP ≤ 5,
    hydrogen bond donors ≤ 5, hydrogen bond acceptors ≤ 10, rotatable bonds ≤ 10).

    Returns a DataFrame with columns: `smiles`, `prediction`, `in_AD`, and
    additional property columns used for filtering.
    """
    records: List[Dict[str, object]] = []
    chooser = rdMolStandardize.LargestFragmentChooser()
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = chooser.choose(mol)
            Chem.SanitizeMol(mol)
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[#6]')):
                continue
            can_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            # Features
            ecfp = _compute_ecfp4_fingerprint(mol)
            maccs = _compute_maccs_fingerprint(mol)
            desc = _compute_rdkit_descriptors(mol)
            if np.isnan(desc).any():
                continue
            X = np.concatenate([ecfp, maccs, desc])
            # Applicability domain check
            sims = BulkTanimotoSimilarity(ecfp, list(train_fps))
            sims = np.sort(np.array(sims))[-k:]
            mean_sim = sims.mean()
            in_ad = mean_sim >= ad_threshold
            # Prediction
            pred = float(model.predict(X.reshape(1, -1))[0])
            # Compute drug‑likeness properties
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            rot = Descriptors.NumRotatableBonds(mol)
            # Filter
            if druglike:
                if (mw > 500 or logp > 5 or hbd > 5 or hba > 10 or rot > 10):
                    continue
            records.append({
                'smiles': can_smi,
                'prediction': pred,
                'mean_similarity': mean_sim,
                'in_AD': in_ad,
                'MolWt': mw,
                'MolLogP': logp,
                'HBD': hbd,
                'HBA': hba,
                'RotatableBonds': rot,
            })
        except Exception:
            continue
    screen_df = pd.DataFrame(records)
    # Sort by predicted potency (higher pIC50 implies more potent)
    screen_df = screen_df.sort_values(by='prediction', ascending=False).reset_index(drop=True)
    return screen_df
