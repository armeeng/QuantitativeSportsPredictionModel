#!/usr/bin/env python3
import re
import sys
import os
from Model import MLModel
from TestModel import TestModel

# --- Path Correction (if your script structure requires it) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def build_model_name(model_type: str, column: str, query: str) -> str:
    # 1) model abbreviation
    abbrs = {
        'linear_regression': 'lr', 'logistic_regression': 'lo',
        'random_forest':     'rf', 'random_forest_regressor': 'rf', 'random_forest_classifier': 'rf',
        'xgboost':           'xgb', 'xgboost_regressor': 'xgb', 'xgboost_classifier': 'xgb',
        'neural_network':    'nn', 'mlp': 'nn', 'mlp_regressor': 'nn', 'mlp_classifier': 'nn'
    }
    m_abbr = abbrs.get(model_type, model_type[:2])

    # 2) column part
    col_part = 'norm' if column == 'normalized_stats' else 'nonorm'

    # 3) sport
    m = re.search(r"sport\s*=\s*'([^']+)'", query, re.IGNORECASE)
    sport = m.group(1).upper() if m else 'ALL'

    # 4) suffix: slugify the conditions to make it descriptive
    try:
        where = query.split('WHERE',1)[1]
        cond = where.strip().rstrip(';').strip()
        rest = re.sub(rf"sport\s*=\s*'{sport}'\s*(AND\s*)?", '', cond, flags=re.IGNORECASE)
        suffix = re.sub(r'[^0-9A-Za-z<>=]+', '_', rest).strip('_').lower()
        if not suffix: suffix = 'all' # Handle case where only sport is present
    except IndexError:
        suffix = 'all' # No WHERE clause found

    return f"{m_abbr}_{col_part}_{sport}_{suffix}"

def main():
    """
    Train/test a model based on a selected cross-validation fold.
    """
    # ==============================================================================
    # --- CONFIGURATION FOR CROSS-VALIDATION ---
    # ==============================================================================
    SPORT_TO_RUN = 'CBB'

    # CHOOSE WHICH SEASON TO USE AS THE TEST SET.
    # The script will automatically use all other seasons as the training set.
    # Options for NFL: '21-22', '22-23', '23-24', '24-25', '25-26'
    TEST_SEASON_NAME = '25-26'

    # Define the seasons for each sport
    SEASONS = {
        'NFL': [
            ('21-22', '2021-09-08', '2022-02-15'),
            ('22-23', '2022-09-07', '2023-02-15'),
            ('23-24', '2023-09-06', '2024-02-15'),
            ('24-25', '2024-09-04', '2025-02-15'),
            ('25-26', '2025-09-04', '2026-02-15')
        ],
        'NBA': [('21-22', '2021-10-18', '2022-06-20'), ('22-23', '2022-10-17', '2023-06-20'), ('23-24', '2023-10-23', '2024-06-20'), ('24-25', '2024-10-21', '2025-06-20')],
        'CFB': [('21-22', '2021-08-27', '2022-01-15'), ('22-23', '2022-08-26', '2023-01-15'), ('23-24', '2023-08-25', '2024-01-15'), ('24-25', '2024-08-23', '2025-01-15'), ('25-26', '2025-08-23', '2026-01-15')],
        'CBB': [('21-22', '2021-11-08', '2022-04-05'), ('22-23', '2022-11-06', '2023-04-05'), ('23-24', '2023-11-05', '2024-04-10'), ('24-25', '2024-11-04', '2025-04-10'), ('25-26', '2025-04-05', '2026-04-20')],
        'MLB': [('2021', '2021-04-01', '2021-11-03'), ('2022', '2022-04-07', '2022-11-06'), ('2023', '2023-03-30', '2023-11-02'), ('2024', '2024-03-28', '2024-11-03'), ('2025', '2025-03-28', '2025-11-03')]
    }
    # ==============================================================================

    # --- Dynamically build Train/Test queries based on config ---
    test_season_info = next((s for s in SEASONS[SPORT_TO_RUN] if s[0] == TEST_SEASON_NAME), None)

    if not test_season_info:
        raise ValueError(f"Season '{TEST_SEASON_NAME}' not found for sport '{SPORT_TO_RUN}'. Please check your configuration.")

    _, test_start_date, test_end_date = test_season_info

    TRAIN_QUERY = (
        f"SELECT * FROM games "
        f"WHERE sport = '{SPORT_TO_RUN}' AND (date < '{test_start_date}' OR date > '{test_end_date}') "
        f"ORDER BY date ASC;"
    )

    TEST_QUERY = (
        f"SELECT * FROM games "
        f"WHERE sport = '{SPORT_TO_RUN}' AND date >= '{test_start_date}' AND date <= '{test_end_date}' "
        f"ORDER BY date ASC;"
    )

    print(f"--- Running Analysis ---")
    print(f"Sport: {SPORT_TO_RUN}")
    print(f"Test Season: {TEST_SEASON_NAME} ({test_start_date} to {test_end_date})")
    print("Training on all other available seasons.")
    print("-" * 24)

    # ── Feature definitions ────────────────────────────────
    num_feat = [72, 248, 249, 250, 253, 255, 760, 762, 778, 927]
    cat_feat = []

    # ── Model & data configuration ─────────────────────────
    MODEL_TYPE = "logistic_regression"
    COLUMN = "stats"

    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ──────────────────────────────────────────────
    model = MLModel(
        MODEL_NAME,
        MODEL_TYPE,
        column=COLUMN,
        hyperparameter_tuning=False,
        tuning_n_iter=100,
        random_state=130,
        numerical_feature_indices=num_feat,
        categorical_feature_names=cat_feat,
        include_market_spread=False,
        include_market_total=False,
        feature_engineering_mode='differential',
        calibrate_model=False,
        calibration_method='sigmoid',
        calibration_split_size=0.2
    )
    model.train(TRAIN_QUERY, TEST_QUERY)

if __name__ == "__main__":
    main()
