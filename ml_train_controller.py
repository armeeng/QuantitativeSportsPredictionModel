#!/usr/bin/env python3
import re
from Model import MLModel
from TestModel import TestModel

def build_model_name(model_type: str, column: str, query: str) -> str:
    # 1) model abbreviation
    abbrs = {
        'linear_regression': 'lr',
        'random_forest':     'rf',
        'xgboost':           'xgb',
        'neural_network':    'nn'
        #'mlp':               'nn',
        #'mlp_regressor':     'nn',
    }
    m_abbr = abbrs.get(model_type, model_type[:2])

    # 2) column part
    col_part = 'norm' if column == 'normalized_stats' else 'nonorm'

    # 3) sport
    m = re.search(r"sport\s*=\s*'([^']+)'", query, re.IGNORECASE)
    sport = m.group(1).upper() if m else 'ALL'

    # 4) suffix: “all” if only sport is in the WHERE clause, else slug of the rest
    where = query.split('WHERE',1)[1]
    # normalize whitespace & strip trailing ;
    cond = where.strip().rstrip(';').strip()
    # check if it’s exactly just the sport clause
    if re.fullmatch(rf"sport\s*=\s*'{sport}'", cond, re.IGNORECASE):
        suffix = 'all'
    else:
        # remove sport=... then slugify remaining
        rest = re.sub(rf"sport\s*=\s*'{sport}'\s*(AND\s*)?", '', cond, flags=re.IGNORECASE)
        # slugify: letters+digits and other things only, underscores
        suffix = re.sub(r'[^0-9A-Za-z<>=]+', '_', rest).strip('_').lower()

    return f"{m_abbr}_{col_part}_{sport}_{suffix}"

def main():
    """
    Train/test an MLB model.
    """

    # ── Feature definitions ────────────────────────────────
    num_feat = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    cat_feat = [
        "team1_id", "team2_id", "venue_id", "season_type",
        "day", "month", "year", "day_of_week",
    ]

    # ── Model & data configuration ─────────────────────────
    MODEL_TYPE = "logistic_regression"          
    # Options include:
    # ['linear_regression', 'random_forest_regressor', 'xgboost_regressor',
    #  'mlp_regressor', 'knn_regressor', 'svr',
    #  'logistic_regression', 'knn_classifier', 'svc',
    #  'random_forest_classifier', 'xgboost_classifier',
    #  'mlp_classifier', 'gradient_boosting_classifier',
    #  'gaussian_nb', 'random_forest', 'xgboost',
    #  'mlp', 'neural_network', 'svm']

    COLUMN = "stats"  # 'stats' or 'normalized_stats'

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

    # Build a name like 'lr_norm_NBA_all' or
    # 'rf_nonorm_MLB_date_<_2025_05_26'
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
        include_market_spread=True,
        include_market_total=True,
    )
    model.train(TRAIN_QUERY, TEST_QUERY)

if __name__ == "__main__":
    main()
