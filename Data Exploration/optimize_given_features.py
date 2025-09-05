#!/usr/bin/env python3

from ast import pattern
import os
import sys
import re
import io
import numpy as np
from contextlib import redirect_stdout
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. use the right features for your sport (get from univariate feature selector)
# 2. update your train and test query to match the train query used previously and what you'll use in the future

# --- Path Correction & Model Imports -----------------------------------------
# Ensure MLModel and TestModel classes can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from Model import MLModel
    from TestModel import TestModel
except ImportError as e:
    print(f"FATAL: Could not import MLModel or TestModel.")
    print(f"Please ensure Model.py and TestModel.py are in the parent directory: '{parent_dir}'")
    print(f"Details: {e}")
    sys.exit(1)

# --- CONFIGURATION -----------------------------------------------------------

# The full list of candidate features to test.
CANDIDATE_FEATURES = [117, 123, 125, 127, 129, 131, 212, 224, 226, 232, 291, 363, 392, 393, 430, 435, 693, 707, 885, 1015, 1033, 1034, 1036, 1057, 1062, 1064, 1137, 1186, 1190, 1263, 1358, 1359, 1362, 1366, 1370, 1373, 1495, 1498, 1499, 1537, 1540, 1547, 1624, 1715, 1717, 1718, 1727, 1728]

# --- HELPER & PARSING FUNCTIONS ----------------------------------------------

def build_model_name(model_type: str, column: str, query: str) -> str:
    """Helper function to create a descriptive model name."""
    m_abbr = {'logistic_regression': 'lr'}.get(model_type, model_type[:2])
    col_part = 'norm' if column == 'normalized_stats' else 'nonorm'
    m = re.search(r"sport\s*=\s*'([^']+)'", query, re.IGNORECASE)
    sport = m.group(1).upper() if m else 'ALL'
    return f"{m_abbr}_{col_part}_{sport}_opt"  # Simplified for optimizer

def parse_output(output_text: str) -> dict:
    """Uses regular expressions to parse all metrics from the captured text."""
    # This dictionary holds the regular expressions to find each metric.
    REGEX_METRICS = {
        # P-Values: Anchored to the specific "Strategy 1" block under the main header.
        'ev_p_value_ml': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Moneyline\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        'ev_p_value_spread': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Spread\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        'ev_p_value_ou': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Ou\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        
        # Kelly Profit: Anchored to the "Kelly Criterion Simulation" header.
        'kelly_profit_ml': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Moneyline:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
        'kelly_profit_spread': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Spread:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
        'kelly_profit_ou': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Over/Under:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
        
        # Bootstrap PoP: Fixed regex patterns to properly capture each betting strategy section
        'bootstrap_pop_ml': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Moneyline Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
        'bootstrap_pop_spread': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Spread Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
        'bootstrap_pop_ou': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Ou Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
        
        'accuracy_ml': re.compile(r"Model Prediction Accuracy:.*?-\s*Winner Accuracy:\s*([\d\.]+)%", re.DOTALL),
        'accuracy_spread': re.compile(r"Model Prediction Accuracy:.*?-\s*Spread Accuracy:\s*([\d\.]+)%", re.DOTALL),
        'accuracy_ou': re.compile(r"Model Prediction Accuracy:.*?-\s*Over/Under Accuracy:\s*([\d\.]+)%", re.DOTALL),
    }
    
    results = {}
    for key, pattern in REGEX_METRICS.items():
        match = pattern.search(output_text)
        if match:
            value_str = match.group(1).replace(',', '')
            results[key] = float(value_str)
        else:
            results[key] = np.nan
            # Debug: print which metric failed to parse
            print(f"[DEBUG] Failed to parse {key}")
    
    return results

# --- CORE LOGIC: TRAINING & EVALUATION ---------------------------------------

def run_training_for_combination(num_feat: list):
    """
    This function IS your training process. It takes a list of features,
    trains the model, and its .train() call prints the full report from TestModel.
    It does not return anything; its purpose is to print to standard output.
    """
    # --- Model & data configuration ---
    MODEL_TYPE = "logistic_regression"
    COLUMN = "stats"
    TRAIN_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'NFL' AND DATE < '2024-07-10' "
        "ORDER BY date ASC;"
    )
    TEST_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'NFL' AND DATE > '2024-07-10' "
        "ORDER BY date ASC;"
    )
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)
    
    # --- TRAIN ---
    model = MLModel(
        MODEL_NAME,
        MODEL_TYPE,
        column=COLUMN,
        # IMPORTANT: Hyperparameter tuning is turned OFF for feature selection.
        # It would make the process 100x slower. Find the best features first,
        # then tune the final model.
        hyperparameter_tuning=False,
        random_state=130,
        numerical_feature_indices=num_feat,
        categorical_feature_names=[],  # Assuming we only optimize numerical features
        include_market_spread=True,
        include_market_total=True,
        feature_engineering_mode='differential',
        calibrate_model=False,
    )
    # This .train() method will trigger the full printout report via TestModel
    model.train(TRAIN_QUERY, TEST_QUERY)

def evaluate_combination(combination: list) -> dict:
    """
    Runs the training process for a given combination, capturing and parsing its output.
    This is the key function that simulates reading from the terminal.
    """
    output_buffer = io.StringIO()
    metrics = {}
    
    try:
        # Redirect all print() statements during training to our in-memory buffer
        with redirect_stdout(output_buffer):
            run_training_for_combination(combination)
        
        # Once done, get the entire text that was "printed"
        output_text = output_buffer.getvalue()
        
        # Parse the desired metrics from the captured text
        metrics = parse_output(output_text)
        
        # Only print metrics if parsing was successful
        if not all(np.isnan(v) for v in metrics.values()):
            print(f"[SUCCESS] Parsed metrics for combo {combination}: {metrics}")
        
        # --- DEBUGGING STEP ---
        # If parsing fails for all metrics, print the captured text to see why.
        if all(np.isnan(v) for v in metrics.values()):
            print(f"\n[DEBUG] Parsing failed for combo {combination}. Captured output was:")
            print("-" * 40)
            print(output_text)
            print("-" * 40)
        
        return metrics
    
    except Exception as e:
        print(f"\n--- An ERROR occurred during evaluation of combo: {combination} ---")
        print(f"Error: {type(e).__name__}, Message: {e}")
        return {}  # Return an empty dict to signify failure
    finally:
        # CRITICAL: Close all matplotlib figures created by TestModel to prevent a memory leak.
        plt.close('all')

def forward_feature_selection(objective: dict):
    """
    Ranks features individually, then adds them sequentially to find the best combo.
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING OPTIMIZATION FOR: {objective['name']} (Goal: {objective['goal']})")
    print(f"{'='*80}")

    # --- Step 1: Rank all candidate features based on their individual score ---
    print("\n--- Step 1: Ranking all features individually ---")
    feature_scores = []
    is_minimize = objective['goal'] == 'minimize'
    
    pbar_rank = tqdm(CANDIDATE_FEATURES, desc="Evaluating individual features", leave=False)
    for feature in pbar_rank:
        pbar_rank.set_postfix_str(f"Testing feature: [{feature}]")
        metrics = evaluate_combination([feature])  # Test each feature by itself
        score = metrics.get(objective['metric_key'], np.nan)
        if not np.isnan(score):
            feature_scores.append((feature, score))

    if not feature_scores:
        print("\nERROR: Could not get a valid score for any individual feature. Stopping.")
        return [], float('inf') if is_minimize else float('-inf')

    # Sort features based on their individual performance (best to worst)
    feature_scores.sort(key=lambda x: x[1], reverse=not is_minimize)
    
    ranked_features = [f for f, s in feature_scores]
    print(f"\n✅ Feature ranking complete. Best performing feature is {ranked_features[0]}.")
    print(f"   Full ranking: {ranked_features}")

    # --- Step 2: Sequentially add features from the ranked list and find the best combination ---
    print("\n--- Step 2: Building combination by adding features in ranked order ---")
    selected_features = []
    best_overall_combination = []
    best_overall_score = float('inf') if is_minimize else float('-inf')
    rounds_without_improvement = 0

    pbar_build = tqdm(ranked_features, desc="Building best combination", leave=False)
    for i, feature_to_add in enumerate(pbar_build):
        selected_features.append(feature_to_add)
        current_combination = sorted(selected_features) # Keep list sorted for consistency
        
        pbar_build.set_postfix_str(f"Testing combo: {current_combination}")
        metrics = evaluate_combination(current_combination)
        current_score = metrics.get(objective['metric_key'], np.nan)
        
        if np.isnan(current_score):
            print(f"\n[!] Skipping round {i+1} due to invalid score for combo: {current_combination}")
            continue

        print(f"\n -> Round {i+1}/{len(ranked_features)} | Added feature {feature_to_add} | Score: {current_score:.4f}")

        # Determine if the current round's score is an improvement
        is_better = (is_minimize and current_score < best_overall_score) or \
                    (not is_minimize and current_score > best_overall_score)

        if is_better:
            best_overall_score = current_score
            best_overall_combination = list(current_combination)
            rounds_without_improvement = 0  # Reset counter
            print(f" ★★★ NEW OVERALL BEST! Score: {best_overall_score:.4f} with combo {best_overall_combination} ★★★")
        else:
            rounds_without_improvement += 1
            print(f" -> No improvement. Rounds without improvement: {rounds_without_improvement}/3.")

        # Check for the early stopping condition
        if rounds_without_improvement >= 3:
            print(f"\n🛑 Stopping early: Best score has not improved for {rounds_without_improvement} consecutive rounds.")
            break
            
    return best_overall_combination, best_overall_score

# --- SCRIPT ENTRYPOINT ------------------------------------------------------

def main():
    """Defines objectives and orchestrates the entire optimization process."""

    objectives = [
        {'name': 'Kelly Profit (Moneyline)', 'metric_key': 'kelly_profit_ml', 'goal': 'maximize'},
        {'name': '+EV P-Value (Moneyline)', 'metric_key': 'ev_p_value_ml', 'goal': 'minimize'},
        {'name': 'Bootstrap PoP (Moneyline)', 'metric_key': 'bootstrap_pop_ml', 'goal': 'maximize'},
        
        {'name': 'Kelly Profit (Spread)', 'metric_key': 'kelly_profit_spread', 'goal': 'maximize'},
        {'name': '+EV P-Value (Spread)', 'metric_key': 'ev_p_value_spread', 'goal': 'minimize'},
        {'name': 'Bootstrap PoP (Spread)', 'metric_key': 'bootstrap_pop_spread', 'goal': 'maximize'},
        
        {'name': 'Kelly Profit (Over/Under)', 'metric_key': 'kelly_profit_ou', 'goal': 'maximize'},
        {'name': 'EV P-Value (Over/Under)', 'metric_key': 'ev_p_value_ou', 'goal': 'minimize'},
        {'name': 'Bootstrap PoP (Over/Under)', 'metric_key': 'bootstrap_pop_ou', 'goal': 'maximize'},

        {'name': 'Accuracy (Moneyline)', 'metric_key': 'accuracy_ml', 'goal': 'maximize'},
        {'name': 'Accuracy (Spread)', 'metric_key': 'accuracy_spread', 'goal': 'maximize'},
        {'name': 'Accuracy (Over/Under)', 'metric_key': 'accuracy_ou', 'goal': 'maximize'},
    ]
    
    final_results = {}
    for obj in objectives:
        best_combo, best_score = forward_feature_selection(obj)
        final_results[obj['name']] = {'combination': best_combo, 'score': best_score}
    
    # --- Print Final Report ---
    print("\n" + "="*80)
    print(" " * 24 + "--- FINAL OPTIMIZATION RESULTS ---")
    print("="*80)
    for name, result in final_results.items():
        score = result.get('score')
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) and not np.isinf(score) else "N/A"
        print(f"\n🎯 Objective: {name}")
        print(f" - 🏆 Best Combination: {result.get('combination') or 'Not Found'}")
        print(f" - 📈 Best Score: {score_str}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()