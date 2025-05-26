#!/usr/bin/env python3
import sqlite3
import pandas as pd
from MLModel import MLModel

def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME    = "nn_norm_MLB_all"
    PREDICT_QUERY = "SELECT * FROM games WHERE sport = 'MLB' AND date = '2025-05-20';"

    # ── PREDICT ────────────────────────────────────xs
    model = MLModel(MODEL_NAME)      # picks up latest trained model
    results = model.predict(PREDICT_QUERY)

    # ── OUTPUT ─────────────────────────────────────
    for r in results:
        print(f"Game {r['game_id']}: Predicted {r['pred_team1']:.1f} – {r['pred_team2']:.1f}")

if __name__ == "__main__":
    main()
