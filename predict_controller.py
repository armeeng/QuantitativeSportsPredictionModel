#!/usr/bin/env python3
import sqlite3
import pandas as pd
from MLModel import MLModel

def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME    = "rf_norm_NBA_all"
    PREDICT_QUERY = "SELECT * FROM games WHERE sport = 'NBA' AND date = '2025-05-26';"

    # ── PREDICT ────────────────────────────────────xs
    model = MLModel(MODEL_NAME)      # picks up latest trained model
    results = model.predict(PREDICT_QUERY)

    # ── OUTPUT ─────────────────────────────────────
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
