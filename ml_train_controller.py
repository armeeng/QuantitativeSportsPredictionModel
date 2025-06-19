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
    indices = [23, 24, 37, 58, 71, 94, 101, 102, 108, 110, 116, 136, 158, 198, 240, 334, 553, 770, 968, 974, 1043, 1045, 1133, 1134, 1147, 1168, 1181, 1204, 1211, 1212, 1218, 1220, 1226, 1246, 1268, 1308, 1350, 1375, 1444, 1663, 1880, 2078, 2084, 2153, 2155]
    MODEL_TYPE   = "svr"   # ['linear_regression', 'random_forest_regressor', 'xgboost_regressor', 'mlp_regressor', 'knn_regressor', 'svr', 'logistic_regression', 'knn_classifier', 'svc', 'random_forest_classifier', 'xgboost_classifier', 'mlp_classifier', 'gradient_boosting_classifier', 'gaussian_nb', 'random_forest', 'xgboost', 'mlp', 'neural_network', 'svm']
    COLUMN       = "stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'MLB' AND DATE < '2024-12-10';"
    TEST_QUERY   = "SELECT * FROM games WHERE sport = 'MLB' AND DATE > '2024-12-10';"

    # build a name like "lr_norm_NBA_all" or e.g. "rf_nonorm_MLB_date_<_2025_05_26"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN, feature_allowlist=indices)#feature_allowlist=indices
    model.train(TRAIN_QUERY, TEST_QUERY)

if __name__ == "__main__":
    main()
