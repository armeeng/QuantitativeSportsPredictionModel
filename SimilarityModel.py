import sqlite3
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances
from BaseModel import BaseModel

class SimilarityModel(BaseModel):
    def __init__(self,
                 model_name: str,
                 distance_metric: str = "euclidean",
                 column: str = "normalized_stats"):
        super().__init__(model_name)
        self.distance_metric = distance_metric  # 'cosine', 'euclidean', etc.
        self.column = column

    def train(self, *args, **kwargs):
    # No training necessary for similarity model
        return

    def predict(self,
                query: str,
                reference_query: str = None,
                top_n: int = 5,
                custom_indices: list[int] = None,
                use_random_weights: bool = False,
                random_weights: list[float] = None,
                use_internal_weights: bool = False,
                **kwargs):
        """
        Predict using similarity to past games.

        Parameters:
          query             : SQL to select games to predict.
          reference_query   : SQL to select historical (with known scores). 
                              If None, infers from query by adding `AND team1_score IS NOT NULL`.
          top_n             : number of nearest neighbors to use.
          custom_indices    : explicit neighbor ranks to use instead of 0..top_n-1.
          use_random_weights: if True, uses `random_weights` provided.
          random_weights    : list of weights summing to 1 matching len(custom_indices or top_n).
          use_internal_weights: if True, weight ∝ 1/(distance+ε).
        """
        # 1) load target and reference frames
        conn   = sqlite3.connect("sports.db")
        df_tgt = pd.read_sql_query(query, conn)
        df_ref = pd.read_sql_query(reference_query, conn).dropna(subset=["team1_score", "team2_score"])
        conn.close()

        if df_tgt.empty:
            logging.warning("No target games to predict.")
            return []
        if df_ref.empty:
            raise ValueError("No reference games found for similarity.")

        # helper: flatten numbers recursively
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

        # 2) build feature matrices
        def build_matrix(df):
            parsed = df[self.column].apply(json.loads)
            X_list = []
            for js in parsed:
                row = []
                flatten_numbers(js, row)
                X_list.append(row)
            lengths = {len(r) for r in X_list}
            if len(lengths) != 1:
                raise ValueError(f"Inconsistent feature lengths: {lengths}")
            return np.array(X_list, dtype=float)

        X_tgt = build_matrix(df_tgt)
        X_ref = build_matrix(df_ref)

        # 3) compute distances
        if self.distance_metric.lower() == "cosine":
            dist_mat = cosine_distances(X_tgt, X_ref)
        else:
            dist_mat = pairwise_distances(
                X_tgt, X_ref, metric=self.distance_metric
            )

        eps = 1e-8
        results = []
        for i, row in enumerate(dist_mat):
            # row: distances from tgt[i] to each ref[j]
            # sort ascending
            idx_sorted = np.argsort(row)

            # pick neighbor indices
            if custom_indices is not None:
                neigh = [idx_sorted[k] for k in custom_indices]
            else:
                neigh = idx_sorted[:top_n]

            # pick distances and corresponding scores
            dists = row[neigh]
            scores = df_ref.iloc[neigh][["team1_score", "team2_score"]].to_numpy()

            # compute weights
            if use_random_weights:
                if not random_weights or len(random_weights) != len(neigh):
                    raise ValueError("random_weights must match number of neighbors")
                w = np.array(random_weights, dtype=float)
            elif use_internal_weights:
                inv = 1.0 / (dists + eps)
                w = inv / inv.sum()
            else:
                w = np.ones(len(neigh), dtype=float) / len(neigh)

            # weighted average
            pred = w @ scores  # shape (2,)
            results.append({
                "game_id":    df_tgt.iloc[i]["game_id"],
                "team1_id":   df_tgt.iloc[i]["team1_id"],
                "team2_id":   df_tgt.iloc[i]["team2_id"],
                "pred_team1": float(pred[0]),
                "pred_team2": float(pred[1]),
                "neighbors":  list(df_ref.iloc[neigh]["game_id"]),
                "weights":    w.tolist(),
                "distances":  dists.tolist(),
            })

        return results