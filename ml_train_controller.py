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
    # ── CONFIG ─────────────────────────────────────
    MODEL_TYPE   = "logistic_regression"   # linear_regression, random_forest, xgboost, neural_network
    COLUMN       = "normalized_stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'MLB';"

    # build a name like "lr_norm_NBA_all" or e.g. "rf_nonorm_MLB_date_<_2025_05_26"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN, use_random_subset_of_features=True, subset_fraction=0.1)
    model.train(TRAIN_QUERY)

if __name__ == "__main__":
    main()
