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
    features = [2, 4, 10, 17, 24, 25, 46, 47, 52, 57, 71, 74, 81, 93, 100, 102, 124, 128, 136, 146, 150, 157, 171, 240, 248, 249, 250, 257, 266, 269, 296, 305, 307, 322, 372, 373, 374, 386, 393, 400, 402, 406, 420, 423, 428, 442, 443, 444, 461, 463, 464, 465, 471, 479, 517, 524, 671, 680, 703, 719, 720, 724, 744, 757, 769, 776, 823, 825, 842, 843, 846, 847, 876, 880, 919, 924, 970, 974, 981, 1013, 1038, 1048, 1092, 1112, 1114, 1120, 1127, 1134, 1135, 1156, 1157, 1162, 1167, 1181, 1184, 1191, 1203, 1210, 1212, 1234, 1238, 1243, 1246, 1256, 1260, 1266, 1267, 1281, 1350, 1358, 1359, 1360, 1367, 1374, 1376, 1379, 1406, 1415, 1417, 1482, 1483, 1484, 1496, 1503, 1510, 1512, 1516, 1530, 1533, 1538, 1552, 1553, 1554, 1571, 1573, 1574, 1575, 1588, 1627, 1634, 1662, 1666, 1693, 1750, 1781, 1790, 1813, 1829, 1830, 1834, 1854, 1867, 1879, 1886, 1896, 1932, 1933, 1935, 1953, 1956, 1957, 1986, 1990, 2029, 2034, 2080, 2084, 2091, 2094, 2148, 2158, 2202]
    # ── CONFIG ─────────────────────────────────────
    MODEL_TYPE   = "logistic_regression"   # ['linear_regression', 'random_forest_regressor', 'xgboost_regressor', 'mlp_regressor', 'knn_regressor', 'svr', 'logistic_regression', 'knn_classifier', 'svc', 'random_forest_classifier', 'xgboost_classifier', 'mlp_classifier', 'gradient_boosting_classifier', 'gaussian_nb', 'random_forest', 'xgboost', 'mlp', 'neural_network', 'svm']
    COLUMN       = "stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'MLB' AND DATE < '2024-12-10';"
    TEST_QUERY   = "SELECT * FROM games WHERE sport = 'MLB' AND DATE > '2024-12-10';"

    # build a name like "lr_norm_NBA_all" or e.g. "rf_nonorm_MLB_date_<_2025_05_26"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN, feature_allowlist=features)#feature_allowlist=features
    model.train(TRAIN_QUERY, TEST_QUERY)

if __name__ == "__main__":
    main()
