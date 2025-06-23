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
    indices = [2238, 1359, 1358, 1281, 784, 1516, 1977, 1168, 1394, 2034, 1114, 158, 1319, 1039, 461, 468, 2089, 1569, 420, 850, 823, 1224, 1032, 1511, 1504, 1268, 1156, 244, 1895, 1013, 1350, 246, 1165, 1752, 333, 2231, 2149, 615, 1053, 1166, 2, 1135, 1035, 2024, 1433, 1155, 807, 283, 1466, 707, 1541, 381, 684, 513, 407, 261, 1812, 1354, 1, 400, 849, 24, 1374, 1622, 2235, 101, 683, 2212, 1856, 109, 547, 774, 1054, 385, 1184, 260, 1297, 1920, 703, 2084, 847, 1271, 6, 1048, 1919, 1992, 758, 671, 2158, 1915, 2145, 2164, 817, 2165, 218, 65, 979, 1216, 463, 1820]    
    MODEL_TYPE   = "logistic_regression"   # ['linear_regression', 'random_forest_regressor', 'xgboost_regressor', 'mlp_regressor', 'knn_regressor', 'svr', 'logistic_regression', 'knn_classifier', 'svc', 'random_forest_classifier', 'xgboost_classifier', 'mlp_classifier', 'gradient_boosting_classifier', 'gaussian_nb', 'random_forest', 'xgboost', 'mlp', 'neural_network', 'svm']
    COLUMN       = "stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'MLB' AND DATE < '2024-12-10' ORDER BY date ASC;"
    TEST_QUERY   = "SELECT * FROM games WHERE sport = 'MLB' AND DATE > '2024-12-10' ORDER BY date ASC;"

    # build a name like "lr_norm_NBA_all" or e.g. "rf_nonorm_MLB_date_<_2025_05_26"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN, hyperparameter_tuning=True, tuning_n_iter=100, feature_allowlist=indices, random_state=130, calibrate_model = True, calibration_method = 'isotonic')#feature_allowlist=indices
    model.train(TRAIN_QUERY, TEST_QUERY)

if __name__ == "__main__":
    main()
