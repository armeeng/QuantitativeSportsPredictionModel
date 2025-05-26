import pandas as pd
from datetime import datetime, timezone
from MLModel import MLModel

def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME   = "nn_norm_NBA_all"
    MODEL_TYPE   = "neural_network"   # linear_regression, random_forest, xgboost, neural_network
    COLUMN       = "normalized_stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'NBA';"

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN)
    model.train(TRAIN_QUERY)

if __name__ == "__main__":
    main()
