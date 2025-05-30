import os
import json
import sqlite3
import glob
from datetime import datetime, timezone
import numpy as np

import pandas as pd
import joblib

from BaseModel import BaseModel

# new imports for additional model types
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

class MLModel(BaseModel):
    def __init__(self, model_name: str, model_type: str = 'linear_regression',
                 column: str = "normalized_stats"):
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()

    def train(self, query: str, test_size: float = 0.2, random_state: int = 42):
        """
        Train on the rows returned by `query`, using the JSON column named `self.column`
        as inputs, and team1_score/team2_score as the two outputs.
        Splits off `test_size` fraction for evaluation, prints metrics (incl. betting),
        then saves the model.
        """
        # 1) load data
        conn = sqlite3.connect("sports.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = df.dropna(subset=["team1_score", "team2_score"])

        # 2) build feature matrix
        parsed = df[self.column].apply(json.loads)
        def flatten_numbers(obj, out):
            if obj is None:
                out.append(0.0)
            elif isinstance(obj, dict):
                for v in obj.values():
                    flatten_numbers(v, out)
            elif isinstance(obj, list):
                for v in obj:
                    flatten_numbers(v, out)
            elif isinstance(obj, bool):
                out.append(1.0 if obj else 0.0)
            elif isinstance(obj, (int, float)):
                out.append(float(obj))
            else:
                # ignore strings
                pass

        X_list = []
        for js in parsed:
            row_feats = []
            flatten_numbers(js, row_feats)
            X_list.append(row_feats)

        lengths = {len(r) for r in X_list}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent feature lengths: {lengths}")
        X = np.array(X_list, dtype=float)

        # 3) extract targets and betting lines
        y       = df[["team1_score","team2_score"]].to_numpy()
        spread  = df["team1_spread"].to_numpy()    # assume team2_spread = -team1_spread
        total   = df["total_score"].to_numpy()
        ml1     = df["team1_moneyline"].fillna(0).to_numpy()
        ml2     = df["team2_moneyline"].fillna(0).to_numpy()

        # 4) train/test split (keep everything aligned)
        (X_train, X_test,
         y_train, y_test,
         spr_train, spr_test,
         tot_train, tot_test,
         ml1_train, ml1_test,
         ml2_train, ml2_test) = train_test_split(
            X, y, spread, total, ml1, ml2,
            test_size=test_size, random_state=random_state
        )

        # 5) pick & fit model
        if self.model_type == 'linear_regression':
            model = LinearRegression()
        elif self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0)
        elif self.model_type in ('neural_network','mlp','mlp_regressor'):
            model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=0)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")

        model.fit(X_train, y_train)

        # 6) evaluate regression metrics
        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        mae  = mean_absolute_error(y_test, y_pred, multioutput='raw_values')

        print(f"\nRegression metrics on {len(y_test)} held‐out games:")
        print(f"  Team1 → R² {r2[0]:.3f},  RMSE {rmse[0]:.2f},  MAE {mae[0]:.2f}")
        print(f"  Team2 → R² {r2[1]:.3f},  RMSE {rmse[1]:.2f},  MAE {mae[1]:.2f}")

        # 7) compute betting metrics on test set
        actual_margin = y_test[:,0] - y_test[:,1]
        pred_margin   = y_pred[:,0] - y_pred[:,1]

        # Win/Loss accuracy
        win_acc = np.mean((pred_margin>0) == (actual_margin>0))

        # 7) compute betting metrics on test set
        actual_margin = y_test[:, 0] - y_test[:, 1]
        pred_margin   = y_pred[:, 0] - y_pred[:, 1]

        # Win/Loss accuracy (all games)
        win_acc = np.mean((pred_margin > 0) == (actual_margin > 0))

        # coerce spreads and totals to float, invalid→nan
        spr = pd.to_numeric(spr_test, errors="coerce").astype(float)
        tot = pd.to_numeric(tot_test, errors="coerce").astype(float)

        # Spread cover accuracy: only where we have a valid team1_spread
        valid_spread = np.isfinite(spr)
        if valid_spread.any():
            cover_pred   = pred_margin[valid_spread]   > spr[valid_spread]
            cover_actual = actual_margin[valid_spread] > spr[valid_spread]
            spread_acc   = np.mean(cover_pred == cover_actual)
        else:
            spread_acc = float("nan")

        # Over/Under accuracy: only where we have a valid total line
        valid_ou = np.isfinite(tot)
        if valid_ou.any():
            total_pred  = y_pred[:, 0] + y_pred[:, 1]
            over_pred   = total_pred[valid_ou] > tot[valid_ou]
            over_actual = (y_test.sum(axis=1))[valid_ou] > tot[valid_ou]
            ou_acc      = np.mean(over_pred == over_actual)
        else:
            ou_acc = float("nan")

        # Moneyline P/L assuming $1 per game
        # if predicted winner is team1, use team1 odds (ml1_test), else team2 odds
        ml_pnl = []
        for pm, am, m1, m2 in zip(pred_margin, actual_margin, ml1_test, ml2_test):
            pick1 = (pm > 0)
            odds  = m1 if pick1 else m2

            # skip missing odds
            if odds == 0:
                continue

            won = (am > 0) if pick1 else (am < 0)
            if won:
                # positive odds => profit = odds/100; negative => profit = 100/abs(odds)
                profit = (odds / 100) if odds > 0 else (100 / abs(odds))
            else:
                profit = -1.0

            ml_pnl.append(profit)

        total_pnl = sum(ml_pnl)

        print(f"\nBetting metrics on test set:")
        print(f"  Win/Loss accuracy: {win_acc:.3%}")
        print(f"  Spread‐cover accuracy: {spread_acc:.3%}")
        print(f"  Over/Under accuracy: {ou_acc:.3%}")
        print(f"  Moneyline P/L (per \$1 stakes): {total_pnl:.2f} units")

        # 8) save without overwriting
        os.makedirs("models", exist_ok=True)
        base = os.path.join("models", self.model_name)
        path = f"{base}.joblib"
        if os.path.exists(path):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            path = f"{base}_{ts}.joblib"

        joblib.dump({
            "model":       model,
            "model_type":  self.model_type,
            "input_shape": X.shape[1],
            "column":      self.column,
            "query":       query,
            "metrics": {
                "r2":           r2.tolist(),
                "rmse":         rmse.tolist(),
                "mae":          mae.tolist(),
                "win_acc":      win_acc,
                "spread_acc":   spread_acc,
                "ou_acc":       ou_acc,
                "ml_pnl":       total_pnl
            }
        }, path)

        print(f"\nTrained {self.model_name} ({self.model_type}) on "
              f"{len(X_train)} train / {len(X_test)} test; saved to {path}\n")



    def predict(self, query: str):
        """
        Load the latest saved model for this.model_name, run `query` to fetch rows,
        parse the same JSON `column` it was trained on, and return predictions.
        """
        # 1) find latest model file
        pattern = os.path.join("models", f"{self.model_name}*.joblib")
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No model files found matching {pattern}")
        model_path = max(candidates, key=os.path.getmtime)

        info        = joblib.load(model_path)
        model       = info["model"]
        column      = info["column"]
        input_shape = info["input_shape"]

        # 2) load data
        conn = sqlite3.connect("sports.db")
        df   = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return []

        # 3) parse JSON column
        parsed = df[column].apply(json.loads)

        # 4) same flatten_numbers helper as in train()
        def flatten_numbers(obj, out):
            if obj is None:
                out.append(0.0)
            elif isinstance(obj, dict):
                for v in obj.values():
                    flatten_numbers(v, out)
            elif isinstance(obj, list):
                for v in obj:
                    flatten_numbers(v, out)
            elif isinstance(obj, bool):
                out.append(1.0 if obj else 0.0)
            elif isinstance(obj, (int, float)):
                out.append(float(obj))
            else:
                # ignore strings or other types
                pass

        # 5) build feature matrix exactly as train()
        X_list = []
        for js in parsed:
            row_feats = []
            flatten_numbers(js, row_feats)
            X_list.append(row_feats)

        # 6) sanity‐check shape
        lengths = {len(r) for r in X_list}
        if len(lengths) != 1:
            raise ValueError(f"Feature mismatch: inconsistent lengths {lengths}")
        if lengths.pop() != input_shape:
            raise ValueError(f"Feature mismatch: model expects {input_shape}, got {len(X_list[0])}")

        X = np.array(X_list)

        # 7) predict
        preds = model.predict(X)
        return [
            {
              "game_id":    gid,
              "team1_id":   t1,
              "team2_id":   t2,
              "pred_team1": float(p1),
              "pred_team2": float(p2)
            }
            for gid, t1, t2, (p1, p2) in zip(
                df["game_id"],
                df["team1_id"],
                df["team2_id"],
                preds
            )
        ]