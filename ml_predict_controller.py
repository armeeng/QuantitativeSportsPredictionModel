#!/usr/bin/env python3
import sqlite3
import pandas as pd
from Model import MLModel

def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME    = "lo_norm_CBB_all"
    PREDICT_QUERY = "SELECT * FROM games WHERE sport = 'CBB' AND date = '2025-02-08';"

    # ── PREDICT ────────────────────────────────────
    model = MLModel(MODEL_NAME)      # picks up latest trained model
    results = model.predict(PREDICT_QUERY)

    # ── OUTPUT ─────────────────────────────────────
    for r in results:
        print(r)

if __name__ == "__main__":
    main()