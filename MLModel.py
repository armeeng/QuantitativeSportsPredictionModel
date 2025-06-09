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


    def train(self, query: str, test_size: float = 0.5, random_state: int = 42):
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
                # ignore strings or other types
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

        # 3) extract targets and betting lines (fillna→0 for missing odds)
        y        = df[["team1_score", "team2_score"]].to_numpy()
        spread   = pd.to_numeric(df["team1_spread"], errors="coerce").to_numpy()
        total    = pd.to_numeric(df["total_score"], errors="coerce").to_numpy()

        # spread odds: assume team2_spread_odds is provided in the DB
        spr_odds1 = pd.to_numeric(df["team1_spread_odds"].fillna(0), errors="coerce").to_numpy()
        spr_odds2 = pd.to_numeric(df["team2_spread_odds"].fillna(0), errors="coerce").to_numpy()

        # total (over/under) odds:
        over_odds = pd.to_numeric(df["over_odds"].fillna(0), errors="coerce").to_numpy()
        under_odds = pd.to_numeric(df["under_odds"].fillna(0), errors="coerce").to_numpy()

        # moneyline odds
        ml1 = pd.to_numeric(df["team1_moneyline"].fillna(0), errors="coerce").to_numpy()
        ml2 = pd.to_numeric(df["team2_moneyline"].fillna(0), errors="coerce").to_numpy()

        # 4) train/test split (keep everything aligned)
        (X_train, X_test,
         y_train, y_test,
         spr_train, spr_test,
         tot_train, tot_test,
         spr_odds1_train, spr_odds1_test,
         spr_odds2_train, spr_odds2_test,
         over_odds_train, over_odds_test,
         under_odds_train, under_odds_test,
         ml1_train, ml1_test,
         ml2_train, ml2_test) = train_test_split(
            X, y,
            spread, total,
            spr_odds1, spr_odds2,
            over_odds, under_odds,
            ml1, ml2,
            test_size=test_size,
            random_state=random_state
        )

        # 5) pick & fit model
        if self.model_type == 'linear_regression':
            model = LinearRegression()
        elif self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0)
        elif self.model_type in ('neural_network', 'mlp', 'mlp_regressor'):
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0)
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
        actual_margin = y_test[:, 0] - y_test[:, 1]
        pred_margin   = y_pred[:, 0] - y_pred[:, 1]

        # Win/Loss accuracy (all games)
        win_acc = np.mean((pred_margin > 0) == (actual_margin > 0))

        # Spread‐cover accuracy & P/L
        spr = spr_test.astype(float)
        valid_spread = np.isfinite(spr)
        if valid_spread.any():
            actual_adjusted = actual_margin + spr
            # Exclude pushes where actual_adjusted == 0
            non_push_spread = valid_spread & (actual_adjusted != 0)
            if non_push_spread.any():
                # cover
                cover_pred   = (pred_margin[non_push_spread] + spr[non_push_spread]) > 0
                cover_actual = actual_adjusted[non_push_spread] > 0
                spread_acc   = np.mean(cover_pred == cover_actual)
            else:
                spread_acc = float("nan")
        else:
            spread_acc = float("nan")

        # Spread P/L: bet $1 on whichever side the model “covers”
        spread_pnl_list = []
        for i in range(len(pred_margin)):
            if not np.isfinite(spr[i]):
                continue  # no spread line

            am = actual_margin[i]
            pm = pred_margin[i]
            line = spr[i]

            # predicted cover side: if (pm + line) > 0, model takes Team1 +line, else Team2 –line
            pick_team1 = (pm + line) > 0
            # Fetch the correct closing spread odds depending on side:
            odds_side = spr_odds1_test[i] if pick_team1 else spr_odds2_test[i]
            if odds_side == 0:
                continue  # skip missing odds

            # Determine if bet won: 
            # If pick_team1, we need (am + line) > 0. If pick team2, we need (am + line) < 0.
            won = ((am + line) > 0) if pick_team1 else ((am + line) < 0)
            if won:
                # positive American odds => profit = odds/100; negative => profit = 100/abs(odds)
                profit = (odds_side / 100) if odds_side > 0 else (100.0 / abs(odds_side))
            else:
                profit = -1.0
            spread_pnl_list.append(profit)
        spread_pnl = sum(spread_pnl_list)

        # Over/Under accuracy & P/L
        total_line = tot_test.astype(float)
        valid_ou = np.isfinite(total_line)
        if valid_ou.any():
            actual_total = y_test.sum(axis=1)
            non_push_ou = valid_ou & (actual_total != total_line)
            if non_push_ou.any():
                total_pred = y_pred.sum(axis=1)
                over_pred  = total_pred[non_push_ou] > total_line[non_push_ou]
                over_actual = actual_total[non_push_ou] > total_line[non_push_ou]
                ou_acc     = np.mean(over_pred == over_actual)
            else:
                ou_acc = float("nan")
        else:
            ou_acc = float("nan")

        # Over/Under P/L: bet $1 on Over if model’s predicted total > line, else Under
        ou_pnl_list = []
        for i in range(len(pred_margin)):
            if not np.isfinite(total_line[i]):
                continue  # no O/U line

            actual_total = y_test[i, 0] + y_test[i, 1]
            predicted_total = y_pred[i, 0] + y_pred[i, 1]
            line = total_line[i]

            pick_over = predicted_total > line
            odds_side = over_odds_test[i] if pick_over else under_odds_test[i]
            if odds_side == 0:
                continue  # skip missing odds

            if pick_over:
                won = (actual_total > line)
            else:
                won = (actual_total < line)

            if won:
                profit = (odds_side / 100) if odds_side > 0 else (100.0 / abs(odds_side))
            else:
                profit = -1.0
            ou_pnl_list.append(profit)
        ou_pnl = sum(ou_pnl_list)

        # Moneyline P/L assuming $1 per game
        ml_pnl_list = []
        for i in range(len(pred_margin)):
            pm = pred_margin[i]
            am = actual_margin[i]
            m1 = ml1_test[i]
            m2 = ml2_test[i]

            pick1 = (pm > 0)
            odds_side = m1 if pick1 else m2
            if odds_side == 0:
                continue  # skip missing odds

            if pick1:
                won = (am > 0)
            else:
                won = (am < 0)

            if won:
                profit = (odds_side / 100) if odds_side > 0 else (100.0 / abs(odds_side))
            else:
                profit = -1.0
            ml_pnl_list.append(profit)
        total_ml_pnl = sum(ml_pnl_list)

        # 8) print all betting metrics
        print(f"\nBetting metrics on test set ({len(y_test)} games):")
        print(f"  Win/Loss accuracy:    {win_acc:.3%}")
        print(f"  Spread‐cover accuracy:{spread_acc:.3%}")
        print(f"  Over/Under accuracy:   {ou_acc:.3%}")
        print(f"  Moneyline  P/L:       {total_ml_pnl:.2f} units per $1")
        print(f"  Spread   P/L:         {spread_pnl:.2f} units per $1")
        print(f"  Over/Under P/L:       {ou_pnl:.2f} units per $1")

        # 9) save without overwriting
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
                "r2":            r2.tolist(),
                "rmse":          rmse.tolist(),
                "mae":           mae.tolist(),
                "win_acc":       win_acc,
                "spread_acc":    spread_acc,
                "ou_acc":        ou_acc,
                "ml_pnl":        total_ml_pnl,
                "spread_pnl":    spread_pnl,
                "ou_pnl":        ou_pnl
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