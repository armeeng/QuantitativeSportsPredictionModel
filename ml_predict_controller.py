#!/usr/bin/env python3
import sqlite3
import pandas as pd
from Model import MLModel
from TestModel import TestModel

def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME    = "MLB_MONEYLINE"
    PREDICT_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'MLB' AND DATE > '2024-12-10' "
        "ORDER BY date ASC;"
    )

    # ── PREDICT ────────────────────────────────────
    model = MLModel(MODEL_NAME)      # picks up latest trained model
    predictions, y_test, test_odds = model.predict(PREDICT_QUERY)

    # ── OUTPUT ─────────────────────────────────────
    test_evaluator = TestModel(predictions=predictions, y_test=y_test, test_odds=test_odds)
    test_evaluator.display_results()

if __name__ == "__main__":
    main()