import pandas as pd
import numpy as np
import sqlite3
import json
import sys
import os

# Important: Make sure this script is in the same directory as your Model.py
# file so it can be imported correctly.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Model import MLModel

# --- Configuration ---
# The database file path.
DB_PATH = "sports.db"
# The SQL query to fetch all game data for a comprehensive analysis.
GAMES_QUERY = "SELECT * FROM games WHERE sport = 'MLB';"
# The column containing the JSON blob of features.
STATS_COLUMN = "stats"
# Set your desired correlation threshold. Any feature pair with a correlation
# greater than this value will be considered for pruning.
CORRELATION_THRESHOLD = 0.5

def find_features_to_keep(correlation_matrix: pd.DataFrame, threshold: float) -> set:
    """
    Identifies features to remove based on a correlation threshold.

    To minimize the number of features, this function iterates through the correlation
    matrix. If a pair of features (A, B) is found to be highly correlated, it
    keeps the first one (A) and adds the second one (B) to a set of features to be
    removed. This ensures that from any group of highly correlated features, only
    one is retained.

    Args:
        correlation_matrix (pd.DataFrame): A DataFrame containing the absolute
            correlation values between features.
        threshold (float): The correlation value above which features are
            considered highly correlated.

    Returns:
        set: A set of feature names to remove.
    """
    features_to_remove = set()
    cols = correlation_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            # If feature i is already in the remove set, we don't need to check it.
            if cols[i] in features_to_remove:
                continue
            
            # If the correlation is above the threshold, mark the second feature for removal.
            if correlation_matrix.loc[cols[i], cols[j]] > threshold:
                features_to_remove.add(cols[j])
                
    return features_to_remove

def main():
    """
    Main function to load data, analyze feature correlation, and print the
    list of indices for the features to keep.
    """
    print("--- Starting Feature Correlation Analysis ---")

    # 1. Load data from the database
    print(f"Loading data from '{DB_PATH}'...")
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(GAMES_QUERY, conn)
    except sqlite3.OperationalError as e:
        print(f"Error connecting to or reading from the database: {e}")
        return

    if df.empty:
        print("Query returned no data. Aborting analysis.")
        return

    # 2. Extract numerical features using the same logic as the Model class
    # We instantiate a dummy model just to use its helper methods.
    print("Extracting numerical features to ensure alignment with the Model class...")
    dummy_model = MLModel(model_name="dummy", column=STATS_COLUMN)
    numerical_features_np, numerical_feature_names = dummy_model._prepare_numerical_features(df)

    if numerical_features_np.size == 0:
        print("No numerical features were extracted. Aborting.")
        return

    # Create a DataFrame for correlation analysis
    features_df = pd.DataFrame(numerical_features_np, columns=numerical_feature_names)
    total_features = len(features_df.columns)
    print(f"Successfully extracted {total_features} numerical features.")

    # 3. Calculate the correlation matrix
    print("Calculating correlation matrix...")
    corr_matrix = features_df.corr().abs()

    # 4. Find features to remove
    print(f"Identifying features with correlation > {CORRELATION_THRESHOLD}...")
    features_to_remove = find_features_to_keep(corr_matrix, CORRELATION_THRESHOLD)
    
    # 5. Determine the final list of features and their indices to KEEP
    features_to_keep = [name for name in features_df.columns if name not in features_to_remove]
    
    # Map feature names to their original indices
    name_to_index_map = {name: i for i, name in enumerate(numerical_feature_names)}
    indices_to_keep = sorted([name_to_index_map[name] for name in features_to_keep])

    # 6. Print the results
    print("\n--- Correlation Analysis Complete ---")
    print(f"Correlation Threshold: {CORRELATION_THRESHOLD}")
    print(f"Original number of numerical features: {total_features}")
    print(f"Number of features to remove: {len(features_to_remove)}")
    print(f"Final number of features to keep: {len(indices_to_keep)}")
    
    if features_to_remove:
        print("\nFeatures marked for removal:")
        for feature in sorted(list(features_to_remove)):
             print(f" - {feature}")

    print("\n" + "="*80)
    print("Copy the list below and pass it to the 'numerical_feature_indices' parameter in your MLModel.")
    print("="*80)
    print(f"indices_to_keep = {indices_to_keep}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()