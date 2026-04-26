#!/usr/bin/env python3

from ast import pattern
import os
import sys
import re
import io
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Path Correction & Model Imports -----------------------------------------
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

# ==============================================================================
# --- MAIN CONFIGURATION ---
# ==============================================================================
SPORT_TO_RUN = 'CBB'
CANDIDATE_FEATURES = [48, 64, 69, 71, 72, 73, 157, 159, 160, 231, 232, 233, 234, 239, 241, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 282, 284, 285, 355, 360, 675, 760, 761, 762, 763, 764, 769, 771, 778, 779, 780, 781, 782, 922, 927, 928, 929, 930, 931, 948]
SEASONS = {
    'NBA': [('21-22', '2021-10-18', '2022-06-20'), ('22-23', '2022-10-17', '2023-06-20'), ('23-24', '2023-10-23', '2024-06-20'), ('24-25', '2024-10-21', '2025-06-20')],
    'NFL': [('21-22', '2021-09-08', '2022-02-15'), ('22-23', '2022-09-07', '2023-02-15'), ('23-24', '2023-09-06', '2024-02-15'), ('24-25', '2024-09-04', '2025-02-15')],
    'CFB': [('21-22', '2021-08-27', '2022-01-15'), ('22-23', '2022-08-26', '2023-01-15'), ('23-24', '2023-08-25', '2024-01-15'), ('24-25', '2024-08-23', '2025-01-15')],
    'CBB': [('21-22', '2021-11-08', '2022-04-05'), ('22-23', '2022-11-06', '2023-04-05'), ('23-24', '2023-11-05', '2024-04-10'), ('24-25', '2024-11-04', '2025-04-10')],
    'MLB': [('2021', '2021-04-01', '2021-11-03'), ('2022', '2022-04-07', '2022-11-06'), ('2023', '2023-03-30', '2023-11-02'), ('2024', '2024-03-28', '2024-11-03')]
}
# ==============================================================================

# --- HELPER & PARSING FUNCTIONS (Unchanged) ----------------------------------
def build_model_name(model_type: str, column: str, sport: str) -> str:
    m_abbr = {'logistic_regression': 'lr'}.get(model_type, model_type[:2])
    col_part = 'norm' if column == 'normalized_stats' else 'nonorm'
    return f"{m_abbr}_{col_part}_{sport.upper()}_opt"

def parse_output(output_text: str) -> dict:
    REGEX_METRICS = {
        'ev_p_value_ml': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Moneyline\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        'ev_p_value_spread': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Spread\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        'ev_p_value_ou': re.compile(r"Strategy 1: Betting only on \+EV Opportunities.*?-\s*Ou\s*:\s*P-value:\s*([\d\.]+)", re.DOTALL),
        'kelly_profit_ml': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Moneyline:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
        'kelly_profit_spread': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Spread:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
        'kelly_profit_ou': re.compile(r"Kelly Criterion Simulation \(Historical Backtest\):.*?-\s*Over/Under:\s*Profit:\s*\$\s*(-?[\d,]+\.[\d]+)", re.DOTALL),
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
        if match: results[key] = float(match.group(1).replace(',', ''))
        else: results[key] = np.nan
    return results
# -----------------------------------------------------------------------------

# --- CORE LOGIC: TRAINING & EVALUATION (Unchanged) ---------------------------
def run_training_for_combination(num_feat: list, train_query: str, test_query: str):
    MODEL_TYPE, COLUMN = "logistic_regression", "stats"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, SPORT_TO_RUN)
    model = MLModel(
        MODEL_NAME, MODEL_TYPE, column=COLUMN, hyperparameter_tuning=False, random_state=130,
        numerical_feature_indices=num_feat, categorical_feature_names=[],
        include_market_spread=False, include_market_total=False,
        feature_engineering_mode='differential', calibrate_model=False,
    )
    model.train(train_query, test_query)

def evaluate_combination(combination: list) -> dict:
    fold_metrics_list = []
    seasons_to_process = SEASONS[SPORT_TO_RUN]
    
    # NEW: Print statement to show which combination is being tested
    print(f"\n  > Evaluating combo {combination} across {len(seasons_to_process)} folds...")

    for season_name, start_date, end_date in seasons_to_process:
        # NEW: Print statement to show which season is the test set for this fold
        print(f"    - Fold: Training on all seasons EXCEPT '{season_name}', testing on '{season_name}'")
        
        output_buffer = io.StringIO()
        train_query = (f"SELECT * FROM games WHERE sport = '{SPORT_TO_RUN}' AND (date < '{start_date}' OR date > '{end_date}') ORDER BY date ASC;")
        test_query = (f"SELECT * FROM games WHERE sport = '{SPORT_TO_RUN}' AND date >= '{start_date}' AND date <= '{end_date}' ORDER BY date ASC;")
        
        try:
            with redirect_stdout(output_buffer):
                run_training_for_combination(combination, train_query, test_query)
            
            # This captures the metrics for this one fold
            fold_metrics = parse_output(output_buffer.getvalue())
            fold_metrics_list.append(fold_metrics)
            # Optional: print metrics for each fold
            # print(f"      Metrics for this fold: { {k: f'{v:.3f}' for k, v in fold_metrics.items() if not np.isnan(v)} }")

        except Exception as e:
            print(f"      ERROR on this fold: {e}")
            fold_metrics_list.append({key: np.nan for key in parse_output("").keys()})
        finally:
            plt.close('all')

    if not fold_metrics_list: return {}

    # This part calculates the average across all the folds
    average_metrics = pd.DataFrame(fold_metrics_list).mean().to_dict()

    # NEW: Print the final averaged metrics before returning
    print(f"    ✅ Averaged Metrics for {combination}: { {k: f'{v:.3f}' for k, v in average_metrics.items() if not np.isnan(v)} }")

    return average_metrics
# -----------------------------------------------------------------------------

def forward_feature_selection(objective: dict, ranked_features: list, all_objectives: list, all_time_bests: dict):
    """
    Builds combinations based on a pre-sorted feature list and performs global best tracking.
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTING OPTIMIZATION FOR: {objective['name']} (Goal: {objective['goal']})")
    print(f"   Using pre-calculated feature ranking: {ranked_features}")
    print(f"{'='*80}")

    selected_features, best_primary_combination = [], []
    is_minimize_primary = objective['goal'] == 'minimize'
    best_primary_score = float('inf') if is_minimize_primary else float('-inf')
    rounds_without_improvement = 0

    pbar_build = tqdm(ranked_features, desc=f"Building for {objective['name']}", leave=False)
    for i, feature_to_add in enumerate(pbar_build):
        selected_features.append(feature_to_add)
        current_combination = sorted(selected_features)
        
        pbar_build.set_postfix_str(f"Testing combo: {current_combination}")
        avg_metrics = evaluate_combination(current_combination)
        if not avg_metrics: continue

        for any_obj in all_objectives:
            obj_name, metric_key, goal = any_obj['name'], any_obj['metric_key'], any_obj['goal']
            new_score = avg_metrics.get(metric_key, np.nan)
            if np.isnan(new_score): continue
            current_best_score = all_time_bests[obj_name]['score']
            is_new_all_time_best = (goal == 'maximize' and new_score > current_best_score) or \
                                   (goal == 'minimize' and new_score < current_best_score)
            if is_new_all_time_best:
                all_time_bests[obj_name].update({'score': new_score, 'combination': list(current_combination)})
                print(f"    🚀 New All-Time Best for '{obj_name}'! Score: {new_score:.4f} with {current_combination}")

        current_primary_score = avg_metrics.get(objective['metric_key'], np.nan)
        if np.isnan(current_primary_score): continue

        is_better_primary = (is_minimize_primary and current_primary_score < best_primary_score) or \
                            (not is_minimize_primary and current_primary_score > best_primary_score)
        if is_better_primary:
            best_primary_score, best_primary_combination, rounds_without_improvement = current_primary_score, list(current_combination), 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= 3:
            print(f"\n🛑 Stopping early: Primary score has not improved for {rounds_without_improvement} rounds.")
            break
            
    return all_time_bests

def main():
    """Defines objectives, performs a one-time feature evaluation, then orchestrates optimization."""
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
    
    all_time_bests = {obj['name']: {'combination': [], 'score': float('-inf') if obj['goal'] == 'maximize' else float('inf')} for obj in objectives}

    # --- Step 1: Initial Evaluation of All Individual Features (Done Once) ---
    print(f"{'='*80}\n🚀 STEP 1: Performing one-time evaluation of all {len(CANDIDATE_FEATURES)} individual features...\n{'='*80}")
    initial_feature_performance = {}
    pbar_initial = tqdm(CANDIDATE_FEATURES, desc="Evaluating all individual features")
    for feature in pbar_initial:
        avg_metrics = evaluate_combination([feature])
        initial_feature_performance[feature] = avg_metrics
        for obj in objectives:
            obj_name, metric_key, goal = obj['name'], obj['metric_key'], obj['goal']
            score = avg_metrics.get(metric_key)
            if score is not None and not np.isnan(score):
                is_new_best = (goal == 'maximize' and score > all_time_bests[obj_name]['score']) or \
                              (goal == 'minimize' and score < all_time_bests[obj_name]['score'])
                if is_new_best: all_time_bests[obj_name].update({'score': score, 'combination': [feature]})
    print("\n✅ Initial evaluation complete.")

    # --- Step 2: Generate Unique Feature Rankings for Each Objective ---
    objective_rankings = {}
    for obj in objectives:
        metric, goal = obj['metric_key'], obj['goal']
        sorted_features = sorted(
            initial_feature_performance.keys(),
            key=lambda f: initial_feature_performance[f].get(metric, float('inf') if goal == 'minimize' else float('-inf')),
            reverse=(goal == 'maximize')
        )
        objective_rankings[obj['name']] = sorted_features

    # --- Step 3: Run Forward Selection for Each Objective Using Its Unique Ranking ---
    for obj in objectives:
        ranked_features_for_obj = objective_rankings[obj['name']]
        all_time_bests = forward_feature_selection(obj, ranked_features_for_obj, objectives, all_time_bests)
    
    # --- Final Report ---
    print(f"\n{'='*80}\n{' ' * 18}--- FINAL GLOBAL BEST OPTIMIZATION RESULTS ---\n{'='*80}")
    for name, result in all_time_bests.items():
        score_str = f"{result['score']:.4f}" if isinstance(result['score'], (int, float)) and not np.isinf(result['score']) else "N/A"
        print(f"\n🎯 Objective: {name}\n - 🏆 Best Combination Found: {result['combination']}\n - 📈 Best Average Score: {score_str}")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()