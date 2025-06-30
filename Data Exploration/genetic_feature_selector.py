import pandas as pd
import numpy as np
import sqlite3
import os
import sys
import random
import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import brier_score_loss

# --- Path Correction & DEAP Installation Check ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("DEAP library not found. Please install it by running: pip install deap")
    sys.exit(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Model import MLModel

# --- Configuration ---
DB_PATH = os.path.join(parent_dir, "sports.db")
GAMES_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'MLB' AND DATE < '2024-12-10' "
        "ORDER BY date ASC;"
    )
STATS_COLUMN = "stats"
CORRELATION_THRESHOLD = 0.9

# --- Genetic Algorithm Configuration ---
POPULATION_SIZE = 50       # How many feature sets to test in each generation.
N_GENERATIONS = 30         # How many generations to evolve.
CROSSOVER_PROB = 0.7       # Probability of two "parent" solutions combining.
MUTATION_PROB = 0.2        # Probability of a random feature being flipped (on/off).
CV_FOLDS = 3               # Folds for cross-validation within the fitness function.

def find_correlated_features_to_remove(features_df: pd.DataFrame, threshold: float) -> set:
    """Finds redundant numerical features to remove based on a correlation threshold."""
    corr_matrix = features_df.corr().abs()
    features_to_remove = set()
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in features_to_remove or cols[j] in features_to_remove:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                features_to_remove.add(cols[j])
    return features_to_remove

# --- The Fitness Function ---
# This function is the heart of the GA. It tells the algorithm how "good"
# each combination of features (each "individual") is.
def evaluate_features(individual, X_num_pool, X_cat_pool, y, num_feature_names, cat_feature_names):
    """
    Trains a model using the feature subset defined by the individual and
    returns its cross-validated Brier score. The GA will aim to MINIMIZE this score.
    """
    # 1. Decode the individual (binary list) into feature names
    num_mask = np.array(individual[:len(num_feature_names)], dtype=bool)
    cat_mask = np.array(individual[len(num_feature_names):], dtype=bool)

    # Ensure at least one feature is selected to prevent errors
    if not np.any(num_mask) and not np.any(cat_mask):
        return (1.0,) # Return worst possible score if no features are selected

    # 2. Create the feature matrix for this individual
    selected_num_features = X_num_pool[:, num_mask]
    selected_cat_features = X_cat_pool[:, cat_mask]
    
    X_subset = np.hstack([selected_num_features, selected_cat_features])

    # 3. Evaluate the feature set using cross-validation
    estimator = lgb.LGBMClassifier(objective='binary', random_state=42, verbosity=-1)
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS)
    
    brier_scores = []
    for train_idx, test_idx in cv_strategy.split(X_subset, y):
        X_train, X_test = X_subset[train_idx], X_subset[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        estimator.fit(X_train, y_train)
        probs = estimator.predict_proba(X_test)[:, 1]
        score = brier_score_loss(y_test, probs)
        brier_scores.append(score)
        
    # Return the average Brier score. The comma is required by DEAP for single-objective optimization.
    return (np.mean(brier_scores),)

def main():
    print("--- Starting Genetic Algorithm Feature Optimizer ---")
    print("This process is computationally intensive. Please be patient.")

    # 1. Load and prepare data
    print("\n[Step 1/3] Loading and Preparing Data...")
    df = pd.read_sql_query(GAMES_QUERY, sqlite3.connect(DB_PATH))
    dummy_model = MLModel(model_name="dummy", column=STATS_COLUMN, feature_engineering_mode='differential')
    X_num, all_numerical_names = dummy_model._prepare_numerical_features(df)
    X_cat_df = dummy_model._extract_categorical_features(df, MLModel._DEFAULT_CATEGORICAL_FEATURES)
    y = (df["team1_score"] > df["team2_score"]).astype(int).to_numpy()
    
    # Prune correlated numerical features first to reduce the search space
    df_num = pd.DataFrame(X_num, columns=all_numerical_names)
    correlated_to_remove = find_correlated_features_to_remove(df_num, CORRELATION_THRESHOLD)
    num_pool_names = [name for name in all_numerical_names if name not in correlated_to_remove]
    X_num_pool = df_num[num_pool_names].to_numpy()
    
    # Use all categorical features as the pool for the GA
    cat_pool_names = list(X_cat_df.columns)
    cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_cat_pool = cat_encoder.fit_transform(X_cat_df)

    n_total_features = len(num_pool_names) + len(cat_pool_names)
    print(f"Data prepared. Optimizing from a pool of {len(num_pool_names)} numerical and {len(cat_pool_names)} categorical features.")

    # 2. Set up the Genetic Algorithm with DEAP
    print("\n[Step 2/3] Setting up Genetic Algorithm...")
    # We want to MINIMIZE the Brier score, so weights are -1.0
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # An individual is a list of 0s and 1s
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_total_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the genetic operators
    toolbox.register("evaluate", evaluate_features, X_num_pool=X_num_pool, X_cat_pool=X_cat_pool, y=y, num_feature_names=num_pool_names, cat_feature_names=cat_pool_names)
    toolbox.register("mate", tools.cxTwoPoint) # Crossover
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Mutation
    toolbox.register("select", tools.selTournament, tournsize=3) # Selection

    # 3. Run the evolution
    print("\n[Step 3/3] Running evolution... This is the longest step.")
    population = toolbox.population(n=POPULATION_SIZE)
    hall_of_fame = tools.HallOfFame(1) # Store the single best individual found
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # The eaSimple algorithm is a standard evolutionary loop
    algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_PROB, mutpb=MUTATION_PROB, 
                        ngen=N_GENERATIONS, stats=stats, halloffame=hall_of_fame, verbose=True)

    # --- Final Results ---
    print("\n" + "="*80)
    print("      >>> GENETIC OPTIMIZATION COMPLETE <<<")
    print("="*80)
    
    best_individual = hall_of_fame[0]
    best_fitness = best_individual.fitness.values[0]
    
    # Decode the best individual to get the final feature lists
    num_mask = np.array(best_individual[:len(num_pool_names)], dtype=bool)
    cat_mask = np.array(best_individual[len(num_pool_names):], dtype=bool)
    
    final_numerical_names = [name for name, selected in zip(num_pool_names, num_mask) if selected]
    final_categorical_names = [name for name, selected in zip(cat_pool_names, cat_mask) if selected]

    # Map numerical names back to their original, pre-pruning indices
    original_num_name_map = {name: i for i, name in enumerate(all_numerical_names)}
    final_numerical_indices = sorted([original_num_name_map[name] for name in final_numerical_names])
    
    print(f"\nBest Brier Score Found: {best_fitness:.5f}")
    print(f"Optimal feature set contains {len(final_numerical_indices)} numerical and {len(final_categorical_names)} categorical features.")
    
    print("\nUse the following lists as parameters for your MLModel instance:\n")
    print("# Best numerical feature indices (for differential mode):")
    print(f"numerical_feature_indices = {final_numerical_indices}\n")
    print("# Best categorical feature names:")
    print(f"categorical_feature_names = {final_categorical_names}\n")
    print("="*80)

if __name__ == "__main__":
    main()