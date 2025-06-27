import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, SequentialFeatureSelector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import brier_score_loss

# --- Path Correction ---
# This ensures the script can find the 'Model.py' file in the parent directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model import MLModel

# --- Configuration ---
DB_PATH = os.path.join(parent_dir, "sports.db")
GAMES_QUERY = "SELECT * FROM games WHERE team1_score IS NOT NULL AND team2_score IS NOT NULL"
STATS_COLUMN = "normalized_stats"
CORRELATION_THRESHOLD = 0.9

# --- Pipeline Configuration ---
# Stage 1: Initial Filtering using Mutual Information.
# Select the top N% of numerical and categorical features independently.
# This provides a major reduction in the search space for the next, more expensive stage.
TOP_PERCENTILE_TO_KEEP = 40 

# Stage 3: Sequential Feature Selection (SFS) Configuration
# This is the core of the optimization. It will iteratively test features.
# 'forward' starts with 0 features and adds them; 'backward' starts with all and removes them.
# Forward is generally faster.
SFS_DIRECTION = 'forward' 
# The number of cross-validation folds to use when evaluating each feature subset.
SFS_CV_FOLDS = 3 

# --- Helper Functions ---
def find_correlated_features_to_remove(features_df: pd.DataFrame, threshold: float) -> set:
    """Finds redundant numerical features to remove based on a correlation threshold."""
    corr_matrix = features_df.corr().abs()
    features_to_remove = set()
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in features_to_remove:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                features_to_remove.add(cols[j])
    return features_to_remove

def main():
    """
    Main function to run the ultimate, multi-stage feature optimization pipeline.
    """
    print("--- Starting Ultimate Feature Optimizer ---")
    print("This process is computationally intensive and may take a very long time.")

    # 1. Load and prepare data
    print(f"\n[Step 1/5] Loading data from '{DB_PATH}'...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(GAMES_QUERY, conn)
    except sqlite3.OperationalError as e:
        print(f"FATAL: Error loading data: {e}")
        return

    dummy_model = MLModel(model_name="dummy", column=STATS_COLUMN)
    
    # Extract ALL numerical and categorical features separately
    X_num, all_numerical_names = dummy_model._prepare_numerical_features(df)
    X_cat_df = dummy_model._extract_categorical_features(df, MLModel._DEFAULT_CATEGORICAL_FEATURES)
    y = (df["team1_score"] > df["team2_score"]).astype(int)

    # Encode categorical features for analysis
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat = cat_encoder.fit_transform(X_cat_df)

    print(f"Successfully loaded {X_num.shape[1]} numerical and {X_cat.shape[1]} categorical features.")

    # --- STAGE 1: Broad Filtering with Mutual Information ---
    print(f"\n[Step 2/5] Filtering to top {TOP_PERCENTILE_TO_KEEP}% of features using Mutual Information...")
    
    # Filter numerical features
    k_num = int((TOP_PERCENTILE_TO_KEEP / 100) * X_num.shape[1])
    mi_selector_num = SelectKBest(mutual_info_classif, k=k_num)
    mi_selector_num.fit(X_num, y)
    num_indices_stage1 = mi_selector_num.get_support(indices=True)
    
    # Filter categorical features
    k_cat = int((TOP_PERCENTILE_TO_KEEP / 100) * X_cat.shape[1])
    # Ensure we keep at least one categorical feature if k_cat becomes 0
    k_cat = max(1, k_cat) 
    mi_selector_cat = SelectKBest(mutual_info_classif, k=k_cat)
    mi_selector_cat.fit(X_cat, y)
    cat_indices_stage1 = mi_selector_cat.get_support(indices=True)

    # Create the intermediate feature sets
    X_num_stage1 = X_num[:, num_indices_stage1]
    X_cat_stage1 = X_cat[:, cat_indices_stage1]
    
    num_names_stage1 = [all_numerical_names[i] for i in num_indices_stage1]
    cat_names_stage1 = [X_cat_df.columns[i] for i in cat_indices_stage1]

    print(f"Reduced to {len(num_names_stage1)} numerical and {len(cat_names_stage1)} categorical features.")

    # --- STAGE 2: Correlation Pruning on Numerical Features ---
    print("\n[Step 3/5] Pruning highly correlated numerical features...")
    df_num_stage1 = pd.DataFrame(X_num_stage1, columns=num_names_stage1)
    correlated_to_remove = find_correlated_features_to_remove(df_num_stage1, CORRELATION_THRESHOLD)
    
    # Get the final list of numerical features after pruning
    num_names_stage2 = [name for name in num_names_stage1 if name not in correlated_to_remove]
    df_num_stage2 = df_num_stage1[num_names_stage2]
    
    # Categorical features pass through unchanged
    df_cat_stage2 = pd.DataFrame(X_cat_stage1, columns=cat_names_stage1)
    
    print(f"Removed {len(correlated_to_remove)} correlated features. {len(num_names_stage2)} numerical features remain.")

    # Combine the pruned numerical and filtered categorical features for the final selection stage
    X_final_pool = pd.concat([df_num_stage2, df_cat_stage2], axis=1)

    # --- STAGE 3: Sequential Feature Selection for Calibration ---
    print("\n[Step 4/5] Running Sequential Feature Selector to optimize for model calibration...")
    print(f"This is the longest step and will test feature combinations using a {SFS_DIRECTION} search.")

    estimator = lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1)
    cv_strategy = StratifiedKFold(n_splits=SFS_CV_FOLDS)
    
    # Define a custom scorer that SequentialFeatureSelector can MINIMIZE.
    # We use Brier Score Loss, where lower is better.
    def brier_scorer(model, X, y):
        probabilities = model.predict_proba(X)[:, 1]
        return -brier_score_loss(y, probabilities) # Negative because SFS maximizes score

    sfs = SequentialFeatureSelector(
        estimator,
        n_features_to_select='auto', # Let SFS decide the best number based on score plateau
        tol=0.001, # Stop if score doesn't improve by this much
        direction=SFS_DIRECTION,
        scoring=brier_scorer,
        cv=cv_strategy,
        n_jobs=-1
    )

    sfs.fit(X_final_pool, y)
    
    # --- Final Results ---
    print("\n[Step 5/5] Finalizing results...")
    
    final_feature_names = list(X_final_pool.columns[sfs.get_support()])
    
    # Separate the final list back into numerical and categorical
    final_numerical_features = [name for name in final_feature_names if name in num_names_stage2]
    final_categorical_features = [name for name in final_feature_names if name in cat_names_stage1]

    # Map numerical feature names back to their original indices
    original_num_name_map = {name: i for i, name in enumerate(all_numerical_names)}
    final_numerical_indices = sorted([original_num_name_map[name] for name in final_numerical_features])
    
    print("\n" + "="*80)
    print("      >>> OPTIMAL FEATURE SET FOR MODEL CALIBRATION <<<")
    print("="*80)
    print(f"\nFound an optimal set of {len(final_feature_names)} features ({len(final_numerical_features)} numerical, {len(final_categorical_features)} categorical).")
    print("\nUse the following two lists as parameters for your MLModel instance:\n")
    
    print("# Best numerical feature indices:")
    print(f"numerical_feature_indices = {final_numerical_indices}\n")
    
    print("# Best categorical feature names:")
    print(f"categorical_feature_names = {final_categorical_features}\n")
    print("="*80)

if __name__ == "__main__":
    main()