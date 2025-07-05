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
CANDIDATE_FEATURES = [
    77, 246, 292, 318, 873, 874, 875, 876, 942, 963, 1036, 1038,
    1041, 1046, 1051, 1053, 1056
]

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
        'kelly_profit_ml': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Moneyline:.*?Profit:\s*\$(-?[\d,\.]+)", re.DOTALL),
        'kelly_profit_spread': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Spread:.*?Profit:\s*\$(-?[\d,\.]+)", re.DOTALL),
        'kelly_profit_ou': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Over/Under:.*?Profit:\s*\$(-?[\d,\.]+)", re.DOTALL),
        
        # Bootstrap PoP: Fixed regex patterns to properly capture each betting strategy section
        'bootstrap_pop_ml': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Moneyline Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
        'bootstrap_pop_spread': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Spread Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
        'bootstrap_pop_ou': re.compile(r"---\s*Bootstrap Simulation Results\s*---.*?Ou Betting Strategy:.*?Probability of Profit:\s*([\d\.]+)%", re.DOTALL),
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
        "WHERE sport = 'MLB' AND DATE < '2024-12-10' "
        "ORDER BY date ASC;"
    )
    TEST_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'MLB' AND DATE > '2024-12-10' "
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
    """Main optimizer loop that implements Forward Feature Selection."""
    print(f"\n{'='*80}")
    print(f"🚀 STARTING OPTIMIZATION FOR: {objective['name']} (Goal: {objective['goal']})")
    print(f"{'='*80}")
    
    selected_features = []
    best_overall_combination = []
    best_overall_score = float('inf') if objective['goal'] == 'minimize' else float('-inf')
    
    for i in range(len(CANDIDATE_FEATURES)):
        remaining = [f for f in CANDIDATE_FEATURES if f not in selected_features]
        if not remaining:
            break
        
        best_feature_this_round = None
        score_this_round = None
        
        pbar = tqdm(remaining, desc=f"Round {i+1}/{len(CANDIDATE_FEATURES)}", leave=False)
        for feature in pbar:
            current_combination = sorted(selected_features + [feature])
            pbar.set_postfix_str(f"Testing combo: {current_combination}")
            
            metrics = evaluate_combination(current_combination)
            current_score = metrics.get(objective['metric_key'], np.nan)
            
            if np.isnan(current_score):
                continue
            
            if score_this_round is None or \
               (objective['goal'] == 'minimize' and current_score < score_this_round) or \
               (objective['goal'] == 'maximize' and current_score > score_this_round):
                score_this_round = current_score
                best_feature_this_round = feature
        
        if best_feature_this_round is None:
            print(f"\n -> Round {i+1} found no valid results or improvements. Stopping.")
            break
        
        selected_features.append(best_feature_this_round)
        selected_features.sort()
        print(f"\n -> Round {i+1} complete. Added feature {best_feature_this_round}. Best score this round: {score_this_round:.4f}")
        
        if (objective['goal'] == 'minimize' and score_this_round < best_overall_score) or \
           (objective['goal'] == 'maximize' and score_this_round > best_overall_score):
            best_overall_score = score_this_round
            best_overall_combination = list(selected_features)
            print(f" ★★★ NEW OVERALL BEST! Combination: {best_overall_combination}, Score: {best_overall_score:.4f} ★★★")
    
    return best_overall_combination, best_overall_score

# --- SCRIPT ENTRYPOINT ------------------------------------------------------

def main():
    """Defines objectives and orchestrates the entire optimization process."""
    # Define all optimization objectives you want to run.
    # To run fewer objectives for a quicker test, just comment out lines here.
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