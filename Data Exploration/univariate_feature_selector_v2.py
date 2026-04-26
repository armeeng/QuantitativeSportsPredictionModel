import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# --- Path Correction ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Model import MLModel

# ==============================================================================
# --- MAIN CONFIGURATION ---
# ==============================================================================
DB_PATH = os.path.join(parent_dir, "sports.db")
STATS_COLUMN = "stats"
TOP_N_FEATURES_TO_LIST = 50

# 1. CHOOSE THE SPORT TO RUN THE ANALYSIS ON
SPORT_TO_RUN = 'CBB' # Options: 'NBA', 'NFL', 'CFB', 'CBB', 'MLB'

# 2. CHOOSE THE TARGET VARIABLE
# Options: 'team1_wins', 'team1_covers', 'over'
TARGET_VARIABLE = 'team1_wins' 

# 3. DEFINE THE SEASONS FOR EACH SPORT
# You can adjust these date ranges as needed.
SEASONS = {
    'NBA': [
        ('21-22', '2021-10-18', '2022-06-20'),
        ('22-23', '2022-10-17', '2023-06-20'),
        ('23-24', '2023-10-23', '2024-06-20'),
        ('24-25', '2024-10-21', '2025-06-20')
    ],
    'NFL': [
        ('21-22', '2021-09-08', '2022-02-15'),
        ('22-23', '2022-09-07', '2023-02-15'),
        ('23-24', '2023-09-06', '2024-02-15'),
        ('24-25', '2024-09-04', '2025-02-15')
    ],
    'CFB': [
        ('21-22', '2021-08-27', '2022-01-15'),
        ('22-23', '2022-08-26', '2023-01-15'),
        ('23-24', '2023-08-25', '2024-01-15'),
        ('24-25', '2024-08-23', '2025-01-15')
    ],
    'CBB': [
        ('21-22', '2021-11-08', '2022-04-05'),
        ('22-23', '2022-11-06', '2023-04-05'),
        ('23-24', '2023-11-05', '2024-04-10'),
        ('24-25', '2024-11-04', '2025-04-10')
    ],
    'MLB': [
        ('2021', '2021-04-01', '2021-11-03'),
        ('2022', '2022-04-07', '2022-11-06'),
        ('2023', '2023-03-30', '2023-11-02'),
        ('2024', '2024-03-28', '2024-11-03')
    ]
}
# ==============================================================================


def run_univariate_feature_test(X: np.ndarray, y: pd.Series, feature_names: list, n_splits: int = 5):
    """
    Tests each feature individually using cross-validation and ranks them by Brier score.
    (This function remains unchanged from your original)
    """
    print(f"\n[INFO] Starting univariate feature testing with {n_splits}-fold cross-validation...")
    print(f"[INFO] This will train and evaluate a model for each of the {len(feature_names)} features.")

    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_performance = []
    
    baseline_prob = y.mean()
    baseline_brier = brier_score_loss(y, np.full(len(y), baseline_prob))
    print(f"      > Baseline Brier Score (predicting average): {baseline_brier:.5f}")


    for i, feature_name in enumerate(feature_names):
        X_single_feature = X[:, i].reshape(-1, 1)
        
        fold_scores = []
        for train_idx, val_idx in cv_strategy.split(X_single_feature, y):
            X_train, X_val = X_single_feature[train_idx], X_single_feature[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create a pipeline to scale data before fitting the logistic regression
            #model = lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1)
            model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
            # model = xgb.XGBClassifier(
            #     objective='binary:logistic',  # Use 'binary:logistic' for probabilities
            #     random_state=42,              # For reproducibility, matching your original code
            #     eval_metric='logloss'         # Suppresses a warning about the default eval metric
            # )
            model.fit(X_train, y_train)
            
            probabilities = model.predict_proba(X_val)[:, 1]
            score = brier_score_loss(y_val, probabilities)
            fold_scores.append(score)
            
        avg_brier_score = np.mean(fold_scores)
        feature_performance.append({
            'feature_name': feature_name,
            'brier_score': avg_brier_score,
        })
        
        print(f"  Processed feature {i + 1}/{len(feature_names)}: '{feature_name}' -> Brier Score: {avg_brier_score:.5f}")

    results_df = pd.DataFrame(feature_performance).sort_values(by='brier_score', ascending=True)
    return results_df

def main():
    """
    Main function to run the leave-one-season-out feature testing pipeline.
    """
    print("--- Starting Univariate Feature Tester (Leave-One-Season-Out) ---")
    print(f"--- Sport: {SPORT_TO_RUN} | Target: {TARGET_VARIABLE} ---")

    # 1. Load all data for the chosen sport
    print(f"\n[Step 1/4] Loading all '{SPORT_TO_RUN}' data from '{DB_PATH}'...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            # Query all data for the sport, let pandas handle date filtering
            query = f"SELECT * FROM games WHERE sport = '{SPORT_TO_RUN}' ORDER BY date ASC;"
            df_all = pd.read_sql_query(query, conn)
            # Ensure date column is in datetime format for proper filtering
            df_all['date'] = pd.to_datetime(df_all['date'])
    except sqlite3.OperationalError as e:
        print(f"FATAL: Error loading data: {e}")
        return

    # Define the target variable 'y' for the entire dataset
    if TARGET_VARIABLE == 'team1_wins':
        y_all = (df_all["team1_score"] > df_all["team2_score"]).astype(int)
    elif TARGET_VARIABLE == 'team1_covers':
        y_all = (df_all["team1_score"] + df_all["team1_spread"] > df_all["team2_score"]).astype(int)
    elif TARGET_VARIABLE == 'over':
        y_all = (df_all["team1_score"] + df_all["team2_score"] > df_all["total_score"]).astype(int)
    else:
        raise ValueError(f"Invalid TARGET_VARIABLE: '{TARGET_VARIABLE}'")

    print(f"Successfully loaded {len(df_all)} games.")

    # 2. Prepare all features once
    print("\n[Step 2/4] Preparing all features for analysis...")
    dummy_model = MLModel(model_name="dummy", column=STATS_COLUMN, feature_engineering_mode='differential')
    X_num_all, all_numerical_names = dummy_model._prepare_numerical_features(df_all)
    X_cat_df_all = dummy_model._extract_categorical_features(df_all, MLModel._DEFAULT_CATEGORICAL_FEATURES)
    
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    X_cat_all = cat_encoder.fit_transform(X_cat_df_all)
    
    all_features_X = np.hstack((X_num_all, X_cat_all))
    all_feature_names = all_numerical_names + list(X_cat_df_all.columns)
    print(f"Prepared a total of {len(all_feature_names)} features.")

    # 3. Run Leave-One-Season-Out Cross-Validation
    seasons_to_process = SEASONS[SPORT_TO_RUN]
    all_fold_results = []
    print(f"\n[Step 3/4] Starting leave-one-season-out validation across {len(seasons_to_process)} seasons...")

    for i, (season_name, start_date, end_date) in enumerate(seasons_to_process):
        print("\n" + "="*80)
        print(f"      >>> FOLD {i+1}/{len(seasons_to_process)}: Holding out season '{season_name}' ({start_date} to {end_date}) <<<")
        print("="*80)
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Define the indices for the training data (everything EXCEPT the held-out season)
        train_indices = (df_all['date'] < start_date) | (df_all['date'] > end_date)
        
        # Select data for the current training fold
        X_train_fold = all_features_X[train_indices]
        y_train_fold = y_all[train_indices]

        if len(y_train_fold) == 0:
            print(f"WARNING: No training data for this fold. Skipping.")
            continue
            
        print(f"Training on {len(y_train_fold)} games...")
        
        # Run the univariate test on this fold's training data
        ranked_features_df = run_univariate_feature_test(X_train_fold, y_train_fold, all_feature_names, n_splits=5)
        all_fold_results.append(ranked_features_df)

    # 4. Aggregate results and display final ranking
    print("\n[Step 4/4] Aggregating results and finalizing ranks...")
    if not all_fold_results:
        print("FATAL: No results were generated from the validation folds.")
        return

    # Combine results and calculate the mean Brier score for each feature
    combined_results_df = pd.concat(all_fold_results)
    final_ranked_features = combined_results_df.groupby('feature_name')['brier_score'].mean().reset_index()
    final_ranked_features = final_ranked_features.sort_values(by='brier_score', ascending=True)

    print("\n" + "="*80)
    print(" >>> TOP INDIVIDUAL FEATURES RANKED BY AVERAGE BRIER SCORE (ACROSS ALL FOLDS) <<<")
    print("="*80)
    print(f"\nShowing the top {TOP_N_FEATURES_TO_LIST} performing features out of {len(all_feature_names)} tested:\n")
    print(final_ranked_features.head(TOP_N_FEATURES_TO_LIST).to_string(index=False))

    # --- Create actionable output for the MLModel class ---
    top_features = final_ranked_features.head(TOP_N_FEATURES_TO_LIST)
    
    final_numerical_indices = []
    final_categorical_names = []
    
    original_num_name_map = {name: i for i, name in enumerate(all_numerical_names)}
    original_cat_names_set = set(X_cat_df_all.columns)
    
    for _, row in top_features.iterrows():
        feature_name = row['feature_name']
        if feature_name in original_num_name_map:
            final_numerical_indices.append(original_num_name_map[feature_name])
        elif feature_name in original_cat_names_set:
            final_categorical_names.append(feature_name)
            
    final_numerical_indices.sort()
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