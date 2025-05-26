import os
import json
import sqlite3
import glob
from datetime import datetime, timezone

import pandas as pd
import joblib

from BaseModel import BaseModel

# new imports for additional model types
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

class MLModel(BaseModel):
    def __init__(self, model_name: str, model_type: str = 'linear_regression',
                 column: str = "normalized_stats"):
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()

    def train(self, query: str):
        """
        Train on the rows returned by `query`, using the JSON column named `self.column`
        as inputs, and team1_score/team2_score as the two outputs.
        Saves the trained model to models/{model_name}.joblib.
        """
        # 1) load data
        conn = sqlite3.connect("sports.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        df = df.dropna(subset=["team1_score", "team2_score"])

        # parse JSON column
        parsed = df[self.column].apply(json.loads)

        # helper: recursively walk a JSON and collect all numbers in insertion order
        def flatten_numbers(obj, out):
            if isinstance(obj, dict):
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

        # build feature matrix
        X_list = []
        for js in parsed:
            row_feats = []
            flatten_numbers(js, row_feats)
            X_list.append(row_feats)

        # ensure all rows have same length
        lengths = {len(r) for r in X_list}
        if len(lengths) != 1:
            raise ValueError(f"Inconsistent feature lengths in JSONs: {lengths}")
        X = pd.np.array(X_list)  # or np.array

        # 2) targets
        y = df[["team1_score", "team2_score"]].to_numpy()

        # 3) pick & fit model
        if self.model_type == 'linear_regression':
            model = LinearRegression()
        elif self.model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=0)
        elif self.model_type == 'xgboost':
            model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=0)
        elif self.model_type in ('neural_network', 'mlp', 'mlp_regressor'):
            model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=0)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")

        model.fit(X, y)

        # 4) save without overwriting
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
            "query":       query
        }, path)

        print(f"Trained {self.model_name} ({self.model_type}) on {len(df)} examples; saved to {path}")



    def predict(self, query: str):
        """
        Load the latest saved model for this.model_name, run `query` to fetch rows,
        parse the same JSON `column` it was trained on, and return predictions.
        Returns:
            List[Dict]: [
                {"game_id": <id>, "pred_team1": float, "pred_team2": float},
                ...
            ]
        """
        pattern = os.path.join("models", f"{self.model_name}*.joblib")
        candidates = glob.glob(pattern)
        if not candidates:
            raise FileNotFoundError(f"No model files found matching {pattern}")
        model_path = max(candidates, key=os.path.getmtime)

        info  = joblib.load(model_path)
        model = info["model"]
        column = info["column"]
        input_shape = info["input_shape"]

        conn = sqlite3.connect("sports.db")
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return []

        parsed = df[column].apply(json.loads)
        flat   = pd.json_normalize(parsed).fillna(0)
        X = flat.to_numpy()
        if X.shape[1] != input_shape:
            raise ValueError(f"Feature mismatch: model expects {input_shape}, got {X.shape[1]}")

        preds = model.predict(X)
        return [
            {"game_id": gid, "pred_team1": float(p1), "pred_team2": float(p2)}
            for gid, (p1, p2) in zip(df["game_id"], preds)
        ]