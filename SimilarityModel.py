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

    def train(self,
              query: str,
              reference_query: str,
              top_n: int = 5,
              custom_indices: list[int] = None,
              use_random_weights: bool = False,
              random_weights: list[float] = None,
              use_internal_weights: bool = False,
              **kwargs):
        """
        “Train” for a similarity model simply means: run predict(...) on the
        same data, then compute betting metrics (win‐loss, spread cover, O/U, and moneyline P/L).
        Returns a dict of metrics.
        """
        # 1) Run predict(...) to get predictions for each target game
        preds = self.predict(
            query=query,
            reference_query=reference_query,
            top_n=top_n,
            custom_indices=custom_indices,
            use_random_weights=use_random_weights,
            random_weights=random_weights,
            use_internal_weights=use_internal_weights,
            **kwargs
        )

        # 2) Load the “target” DataFrame (so we can compare predicted vs actual)
        conn = sqlite3.connect("sports.db")
        df_tgt = pd.read_sql_query(query, conn).dropna(
            subset=["team1_score", "team2_score"]
        )
        conn.close()

        if df_tgt.empty:
            logging.warning("No target games to evaluate.")
            return {}

        # 3) Build arrays of predicted vs actual, along with lines
        # We'll iterate through preds[], which must be in the same order as df_tgt rows.
        # preds[i]["game_id"] should match df_tgt.iloc[i]["game_id"]. If they ever misalign,
        # you may need to re‐index by game_id. For safety, let's build a map.

        # Map game_id -> (pred_team1, pred_team2, game_id) from preds list
        pred_map = {
            int(d["game_id"]): (d["pred_team1"], d["pred_team2"])
            for d in preds
        }

        actual_scores = []
        predicted_scores = []
        spreads = []
        totals = []
        ml1 = []
        ml2 = []
        game_ids = []

        for idx, row in df_tgt.iterrows():
            gid = int(row["game_id"])
            if gid not in pred_map:
                # If predict(...) somehow skipped this game, skip metrics for it.
                continue
            p1, p2 = pred_map[gid]
            actual_scores.append((float(row["team1_score"]), float(row["team2_score"])))
            predicted_scores.append((p1, p2))
            game_ids.append(gid)

            # get the spread and total lines (could be NULL)
            spr1 = row.get("team1_spread", None)
            # team1_spread is the line for team1 relative to team2
            spreads.append(np.nan if spr1 is None else float(spr1))

            tot = row.get("total_score", None)
            totals.append(np.nan if tot is None else float(tot))

            # moneyline odds (could be NULL)
            m1 = row.get("team1_moneyline", None)
            m2 = row.get("team2_moneyline", None)
            ml1.append(np.nan if m1 is None else float(m1))
            ml2.append(np.nan if m2 is None else float(m2))

        # Convert to arrays
        actual_arr = np.array(actual_scores, dtype=float)      # shape (N,2)
        pred_arr   = np.array(predicted_scores, dtype=float)   # shape (N,2)
        spreads_arr = np.array(spreads, dtype=float)           # shape (N,)
        totals_arr  = np.array(totals, dtype=float)            # shape (N,)
        ml1_arr     = np.array(ml1, dtype=float)               # shape (N,)
        ml2_arr     = np.array(ml2, dtype=float)               # shape (N,)

        # 4) Compute the betting metrics:

        # Win/Loss accuracy (all games we have predictions for)
        actual_margin = actual_arr[:, 0] - actual_arr[:, 1]
        pred_margin   = pred_arr[:, 0] - pred_arr[:, 1]
        win_acc = np.mean((pred_margin > 0) == (actual_margin > 0))

        # Spread‐cover accuracy: only consider rows where spread is finite AND not a push
        valid_spr = np.isfinite(spreads_arr)
        if valid_spr.any():
            # Exclude any actual “pushes” where actual_margin + spread == 0
            actual_adjusted = actual_margin + spreads_arr
            non_push_mask = valid_spr & (actual_adjusted != 0)

            if non_push_mask.any():
                # For the remaining rows, determine whether predicted vs actual cover
                cover_pred   = (pred_margin[non_push_mask] + spreads_arr[non_push_mask]) > 0
                cover_actual = actual_adjusted[non_push_mask] > 0
                spread_acc   = np.mean(cover_pred == cover_actual)
            else:
                spread_acc = float("nan")
        else:
            spread_acc = float("nan")

        # Over/Under accuracy: only consider rows where total line is finite AND not a push
        valid_ou = np.isfinite(totals_arr)
        if valid_ou.any():
            # Exclude pushes where actual_total == total_line
            actual_total = actual_arr.sum(axis=1)
            non_push_ou_mask = valid_ou & (actual_total != totals_arr)

            if non_push_ou_mask.any():
                total_pred   = pred_arr.sum(axis=1)
                over_pred    = total_pred[non_push_ou_mask] > totals_arr[non_push_ou_mask]
                over_actual  = actual_total[non_push_ou_mask] > totals_arr[non_push_ou_mask]
                ou_acc       = np.mean(over_pred == over_actual)
            else:
                ou_acc = float("nan")
        else:
            ou_acc = float("nan")

        # Moneyline P/L (per $1 stake on the favorite)
        # We assume: if predicted margin > 0, we “bet” on team1; else we “bet” on team2.
        # Payout: if you bet $1 on a team at +X moneyline, you win (X/100) if they win;
        # if you bet $1 on a team at –Y moneyline, you must risk $1 to win 100/Y.
        ml_pl_list = []
        for i in range(len(actual_arr)):
            margin = actual_margin[i]
            if margin == 0:
                # treat a tie as no P/L for moneyline
                ml_pl_list.append(0.0)
                continue

            # Determine which side we “bet” on:
            if pred_margin[i] > 0:
                # we bet on team1; check the actual outcome
                if actual_arr[i, 0] > actual_arr[i, 1]:
                    # team1 won: payout based on team1_moneyline
                    ml = ml1_arr[i]
                    if not np.isfinite(ml):
                        pl = 0.0
                    elif ml > 0:
                        # e.g. +150: $1 → win $1.50
                        pl = ml / 100.0
                    else:
                        # e.g. -120: you must risk $1.20 to win $1
                        pl = 1.0 / (abs(ml) / 100.0)
                else:
                    # team1 lost → you lose $1
                    pl = -1.0

            else:
                # we bet on team2
                if actual_arr[i, 1] > actual_arr[i, 0]:
                    ml = ml2_arr[i]
                    if not np.isfinite(ml):
                        pl = 0.0
                    elif ml > 0:
                        pl = ml / 100.0
                    else:
                        pl = 1.0 / (abs(ml) / 100.0)
                else:
                    pl = -1.0

            ml_pl_list.append(pl)

        # Sum up total P/L per $1 stake
        total_ml_pl = np.nansum(ml_pl_list)

        # 5) Return a dictionary of metrics (or you could print them)
        return {
            "n_games_evaluated": len(actual_arr),
            "win_loss_accuracy": float(win_acc),
            "spread_accuracy":   float(spread_acc) if not np.isnan(spread_acc) else None,
            "over_under_accuracy": float(ou_acc)   if not np.isnan(ou_acc)   else None,
            "moneyline_pl": total_ml_pl,  # total P/L if you'd bet $1 on each target game
        }

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
            # row[j] = distance from tgt[i] to ref[j]
            # we need to exclude any ref[j] whose game_id == df_tgt.iloc[i]["game_id"]
            current_id = int(df_tgt.iloc[i]["game_id"])
            # create a boolean mask of allowed reference indices
            mask = df_ref["game_id"].astype(int) != current_id

            # If, after masking, all distances are masked, error:
            if not mask.any():
                raise ValueError(
                    f"After excluding itself, no reference games remain for target game_id {current_id}"
                )

            # Build an array of distances where masked-out entries get +inf
            masked_dist = np.where(mask.values, row, np.inf)

            # sort masked distances ascending
            idx_sorted = np.argsort(masked_dist)

            # pick neighbor indices
            if custom_indices is not None:
                neigh = [idx for k in custom_indices for idx in [idx_sorted[k]]]
            else:
                neigh = idx_sorted[:top_n]

            # pick distances and corresponding scores
            dists = masked_dist[neigh]
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

            # weighted average of neighbor scores
            pred = w @ scores  # shape (2,)

            results.append({
                "game_id":    int(df_tgt.iloc[i]["game_id"]),
                "team1_id":   int(df_tgt.iloc[i]["team1_id"]),
                "team2_id":   int(df_tgt.iloc[i]["team2_id"]),
                "pred_team1": float(pred[0]),
                "pred_team2": float(pred[1]),
                "neighbors":  list(df_ref.iloc[neigh]["game_id"].astype(int)),
                "weights":    w.tolist(),
                "distances":  dists.tolist(),
            })

        return results