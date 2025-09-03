import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import brier_score_loss

# 1. update games query to use your training range and select the right sport
# 2. choose the target (team1 wins, team1 covers, over)

# --- Path Correction ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model import MLModel

# --- Configuration ---
DB_PATH = os.path.join(parent_dir, "sports.db")
GAMES_QUERY = (
    "SELECT * FROM games "
    "WHERE sport = 'CFB' AND DATE < '2024-07-10' "
    "ORDER BY date ASC;"
)
STATS_COLUMN = "stats"
# Number of top-performing single features to output in the final list
TOP_N_FEATURES_TO_LIST = 50

def run_univariate_feature_test(X: np.ndarray, y: pd.Series, feature_names: list, n_splits: int = 5):
    """
    Tests each feature individually using cross-validation and ranks them by Brier score.

    This function iterates through every feature provided, trains a simple LightGBM model
    on just that one feature, and evaluates its performance using the Brier score averaged
    over several cross-validation folds. A lower Brier score is better.

    Args:
        X (np.ndarray): The feature data matrix (samples x features).
        y (pd.Series): The target variable.
        feature_names (list): A list of names corresponding to the columns in X.
        n_splits (int): The number of folds to use for StratifiedKFold cross-validation.

    Returns:
        pd.DataFrame: A DataFrame containing all features, sorted by their
                      average cross-validated Brier score (best to worst).
    """
    print(f"\n[INFO] Starting univariate feature testing with {n_splits}-fold cross-validation...")
    print(f"[INFO] This will train and evaluate a model for each of the {len(feature_names)} features.")

    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_performance = []
    
    # Calculate a baseline Brier score (a model that always predicts the mean)
    # Any feature performing worse than this is not useful on its own.
    baseline_prob = y.mean()
    baseline_brier = brier_score_loss(y, np.full(len(y), baseline_prob))
    print(f"      > Baseline Brier Score (predicting average): {baseline_brier:.5f}")


    for i, feature_name in enumerate(feature_names):
        # We need a 2D array for scikit-learn, so we reshape the single feature column
        X_single_feature = X[:, i].reshape(-1, 1)
        
        fold_scores = []
        # Use cross-validation for a more robust score estimate
        for train_idx, val_idx in cv_strategy.split(X_single_feature, y):
            X_train, X_val = X_single_feature[train_idx], X_single_feature[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1)
            model.fit(X_train, y_train)
            
            # Predict probabilities for the positive class (class 1)
            probabilities = model.predict_proba(X_val)[:, 1]
            score = brier_score_loss(y_val, probabilities)
            fold_scores.append(score)
            
        avg_brier_score = np.mean(fold_scores)
        feature_performance.append({
            'feature_name': feature_name,
            'brier_score': avg_brier_score,
            'original_index': i # We preserve the index from the combined feature list
        })
        
        # Progress update
        print(f"  Processed feature {i + 1}/{len(feature_names)}: '{feature_name}' -> Brier Score: {avg_brier_score:.5f}")

    # Sort results by Brier score in ascending order (lower is better)
    results_df = pd.DataFrame(feature_performance).sort_values(by='brier_score', ascending=True)
    return results_df

def main():
    """
    Main function to run the univariate feature testing pipeline.
    """
    print("--- Starting Univariate Feature Tester ---")

    # 1. Load and prepare data
    print(f"\n[Step 1/3] Loading data from '{DB_PATH}'...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(GAMES_QUERY, conn)
    except sqlite3.OperationalError as e:
        print(f"FATAL: Error loading data: {e}")
        return

    dummy_model = MLModel(model_name="dummy", column=STATS_COLUMN, feature_engineering_mode='differential')
    
    # Extract ALL numerical and categorical features separately
    X_num, all_numerical_names = dummy_model._prepare_numerical_features(df)
    X_cat_df = dummy_model._extract_categorical_features(df, MLModel._DEFAULT_CATEGORICAL_FEATURES)

    # choose whatever you want the features for
    # team1 wins
    #y = (df["team1_score"] > df["team2_score"]).astype(int)
    # team1 covers
    y = (df["team1_score"] + df["team1_spread"] > df["team2_score"]).astype(int)
    # total score goes over the line
    #y = (df["team1_score"] + df["team2_score"] > df["total_score"]).astype(int)

    # Encode categorical features for analysis
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    X_cat = cat_encoder.fit_transform(X_cat_df)

    print(f"Successfully loaded {X_num.shape[1]} numerical and {X_cat.shape[1]} categorical features.")

    # --- STAGE 2: Combine all features and run the univariate test ---
    print("\n[Step 2/3] Preparing to test each feature individually...")
    
    # Combine numerical and categorical features into a single matrix and name list
    # This simplifies the looping process
    all_features_X = np.hstack((X_num, X_cat))
    all_feature_names = all_numerical_names + list(X_cat_df.columns)
    
    # This is the new core function that replaces the old multi-stage pipeline
    ranked_features_df = run_univariate_feature_test(all_features_X, y, all_feature_names, n_splits=5)

    # --- STAGE 3: Display Results ---
    print("\n[Step 3/3] Finalizing results...")
    print("\n" + "="*80)
    print("      >>> TOP INDIVIDUAL FEATURES RANKED BY BRIER SCORE (Lower is Better) <<<")
    print("="*80)
    
    # Display the top N features in a readable table
    print(f"\nShowing the top {TOP_N_FEATURES_TO_LIST} performing features out of {len(all_feature_names)} tested:\n")
    print(ranked_features_df.head(TOP_N_FEATURES_TO_LIST).to_string(index=False))

    # --- Create actionable output for the MLModel class ---
    # We need to map the combined indices back to their original numerical/categorical types
    
    top_features = ranked_features_df.head(TOP_N_FEATURES_TO_LIST)
    
    final_numerical_indices = []
    final_categorical_names = []
    
    # Create maps for quick lookup
    original_num_name_map = {name: i for i, name in enumerate(all_numerical_names)}
    original_cat_names = list(X_cat_df.columns)
    
    for _, row in top_features.iterrows():
        feature_name = row['feature_name']
        if feature_name in original_num_name_map:
            # It's a numerical feature, get its original index
            final_numerical_indices.append(original_num_name_map[feature_name])
        elif feature_name in original_cat_names:
            # It's a categorical feature, get its name
            final_categorical_names.append(feature_name)
            
    final_numerical_indices.sort() # Keep the list sorted for consistency
    final_categorical_names.sort()

    print("\n\n" + "="*80)
    print(f"   >>> PARAMETER LISTS FOR THE TOP {TOP_N_FEATURES_TO_LIST} FEATURES <<<")
    print("="*80)
    print("\nUse the following lists as parameters for your MLModel instance:\n")
    
    print("# Best numerical feature indices:")
    print(f"numerical_feature_indices = {final_numerical_indices}\n")
    
    print("# Best categorical feature names:")
    print(f"categorical_feature_names = {final_categorical_names}\n")
    print("="*80)

if __name__ == "__main__":
    main()

# ==============================================================================
# The original multi-stage pipeline is preserved below for reference.
# You can swap the call in main() back to this function if needed.
# ==============================================================================
"""
def run_original_sfs_pipeline():
    # This function would contain the original code from your script's main(),
    # specifically steps 2, 3, 4, and 5 related to Mutual Information,
    # Correlation Pruning, and Sequential Feature Selection.
    # For brevity, it is not copied here, but you can reconstruct it from your
    # original file if you wish to switch between methods.
    pass
"""