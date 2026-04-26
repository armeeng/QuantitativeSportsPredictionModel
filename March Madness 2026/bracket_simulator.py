#!/usr/bin/env python3
"""
March Madness Bracket Simulator
================================
HOW TO USE:
  1. Fill in the BRACKET dict below with the real team names exactly
     as TeamRankings knows them (same names you use elsewhere in the project).
  2. Make sure your trained CBB model .joblib file is in the models/ directory.
  3. Run:  python march_madness_simulator.py
  4. Stats are fetched LIVE from TeamRankings at runtime, so running on
     March 15 uses March 15 stats for every round.

BRACKET FORMAT:
  Each region is a list of 8 first-round matchups in standard NCAA order:
    [1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15]
  Each matchup is a dict: {"team1": "Name", "seed1": N, "team2": "Name", "seed2": N}
  team1 is always the higher seed (lower number).

OUTPUT LEGEND:
  ✓  = winner of that game        ⚠  = manual override applied
  F: = forward pass probability   R: = reverse pass probability
  The two passes cancel home/away positional bias (see predict_matchup docstring).
  Final probability = average of F and R.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from datetime import date, datetime, timezone

# ─── Make sure project files are importable ───────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from Pregame import Pregame
from Model import MLModel


MODEL_NAME = "lo_nonorm_CBB_date_<_2025_04_05_or_date_>_2026_04_20_order_by_date_asc"


FIRST_FOUR = [
    {"team1": "UMBC",         "seed1": 16, "team2": "Howard",   "seed2": 16, "replaces": "A", "winner": "Howard", "winner_seed": 16},
    {"team1": "Prairie View", "seed1": 16, "team2": "Lehigh",   "seed2": 16, "replaces": "B", "winner": "Prairie View", "winner_seed": 16},
    {"team1": "Texas",        "seed1": 11, "team2": "NC State", "seed2": 11, "replaces": "C", "winner": "Texas", "winner_seed": 11},
    {"team1": "Miami OH",     "seed1": 11, "team2": "SMU",      "seed2": 11, "replaces": "D", "winner": "Miami OH", "winner_seed": 11},
]

BRACKET = {
    "East": [
        {"team1": "Duke",           "seed1": 1,  "team2": "Siena",       "seed2": 16},
        {"team1": "Ohio St",        "seed1": 8,  "team2": "TCU",         "seed2": 9 },
        {"team1": "St John's",      "seed1": 5,  "team2": "N Iowa",      "seed2": 12},
        {"team1": "Kansas",         "seed1": 4,  "team2": "Cal Baptist", "seed2": 13},
        {"team1": "Louisville",     "seed1": 6,  "team2": "S Florida",   "seed2": 11},
        {"team1": "Michigan St",    "seed1": 3,  "team2": "N Dakota St", "seed2": 14},
        {"team1": "UCLA",           "seed1": 7,  "team2": "UCF",         "seed2": 10},
        {"team1": "UConn",          "seed1": 2,  "team2": "Furman",      "seed2": 15},
    ],
    "South": [
        {"team1": "Florida",        "seed1": 1,  "team2": "B",           "seed2": 16},
        {"team1": "Clemson",        "seed1": 8,  "team2": "Iowa",        "seed2": 9 },
        {"team1": "Vanderbilt",     "seed1": 5,  "team2": "McNeese",     "seed2": 12},
        {"team1": "Nebraska",       "seed1": 4,  "team2": "Troy",        "seed2": 13},
        {"team1": "North Carolina", "seed1": 6,  "team2": "VCU",         "seed2": 11},
        {"team1": "Illinois",       "seed1": 3,  "team2": "Penn",        "seed2": 14},
        {"team1": "Saint Mary's",   "seed1": 7,  "team2": "Texas A&M",   "seed2": 10},
        {"team1": "Houston",        "seed1": 2,  "team2": "Idaho",       "seed2": 15},
    ],
    "West": [
        {"team1": "Arizona",        "seed1": 1,  "team2": "LIU",         "seed2": 16},
        {"team1": "Villanova",      "seed1": 8,  "team2": "Utah St",     "seed2": 9 },
        {"team1": "Wisconsin",      "seed1": 5,  "team2": "High Point",  "seed2": 12},
        {"team1": "Arkansas",       "seed1": 4,  "team2": "Hawai'i",     "seed2": 13},
        {"team1": "BYU",            "seed1": 6,  "team2": "C",           "seed2": 11},
        {"team1": "Gonzaga",        "seed1": 3,  "team2": "Kennesaw St", "seed2": 14},
        {"team1": "Miami",          "seed1": 7,  "team2": "Missouri",    "seed2": 10},
        {"team1": "Purdue",         "seed1": 2,  "team2": "Queens",      "seed2": 15},
    ],
    "Midwest": [
        {"team1": "Michigan",       "seed1": 1,  "team2": "A",           "seed2": 16},
        {"team1": "Georgia",        "seed1": 8,  "team2": "Saint Louis", "seed2": 9 },
        {"team1": "Texas Tech",     "seed1": 5,  "team2": "Akron",       "seed2": 12},
        {"team1": "Alabama",        "seed1": 4,  "team2": "Hofstra",     "seed2": 13},
        {"team1": "Tennessee",      "seed1": 6,  "team2": "D",           "seed2": 11},
        {"team1": "Virginia",       "seed1": 3,  "team2": "Wright St",   "seed2": 14},
        {"team1": "Kentucky",       "seed1": 7,  "team2": "Santa Clara", "seed2": 10},
        {"team1": "Iowa St",        "seed1": 2,  "team2": "Tennessee St","seed2": 15},
    ],
}

FINAL_FOUR_PAIRINGS = [("East", "South"), ("West", "Midwest")]

# ==============================================================================
#  INJURY ADJUSTMENTS
#  Multipliers on a team's win probability before renormalizing.
#    < 1.0 weakens the team   > 1.0 strengthens them
#  Leave empty if none: INJURY_ADJUSTMENTS = {}
# ==============================================================================
INJURY_ADJUSTMENTS = {
    # "Duke":    0.85,
    # "Houston": 0.92,
}

# ==============================================================================
#  MANUAL OVERRIDES
#  Force a team to win every game they play in the specified round(s).
#  Valid rounds: "First Four", "Round of 64", "Round of 32", "Sweet 16",
#                "Elite 8", "Final Four", "Championship"
#  Leave empty if none: OVERRIDES = {}
# ==============================================================================
OVERRIDES = {
    #"Ohio St": ["Round of 64", "Round of 32"],
}

# ==============================================================================
#  END OF BRACKET — do not edit below unless you know what you're doing
# ==============================================================================

ROUND_NAMES = [
    "Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship",
]
ROUND_POINTS = [1, 2, 4, 8, 16, 32]

# Lines per R64 slot in the bracket canvas. Must be 4 (gives 32 total rows).
_UNIT = 4


# ==============================================================================
#  OVERRIDE HELPER
# ==============================================================================

def _check_override(t1: str, t2: str, round_name: str):
    if t1 in OVERRIDES and round_name in OVERRIDES[t1]:
        return t1
    if t2 in OVERRIDES and round_name in OVERRIDES[t2]:
        return t2
    return None


# ==============================================================================
#  STATS / DATAFRAME BUILDERS
# ==============================================================================

def build_stats_blob(team1_name, team2_name, t1_stats, t2_stats, game_date):
    epoch = date(1970, 1, 1)
    base = {
        "day": game_date.day, "month": game_date.month, "year": game_date.year,
        "days_since_epoch": (game_date - epoch).days,
        "game_time": 19.0, "day_of_week": game_date.weekday(),
        "team1_id": 0, "team2_id": 0, "venue_id": None,
        "is_neutral": True, "is_conference": False, "season_type": 3, "weather": {},
    }
    raw_t1  = {s: i["raw"]        for s, i in t1_stats.items()}
    raw_t2  = {s: i["raw"]        for s, i in t2_stats.items()}
    norm_t1 = {s: i["normalized"] for s, i in t1_stats.items()}
    norm_t2 = {s: i["normalized"] for s, i in t2_stats.items()}
    return ({**base, "team1_stats": raw_t1,  "team2_stats": raw_t2},
            {**base, "team1_stats": norm_t1, "team2_stats": norm_t2})


def build_game_dataframe(team1_name, team2_name, stats_cache, game_date, column):
    raw_blob, norm_blob = build_stats_blob(
        team1_name, team2_name,
        stats_cache[team1_name], stats_cache[team2_name], game_date)
    row = {
        "game_id": f"mm_{team1_name}_{team2_name}".replace(" ", "_"),
        "date": game_date.isoformat(),
        "team1_id": 0, "team2_id": 0,
        "team1_name": team1_name, "team2_name": team2_name,
        "is_neutral": 1, "is_conference": 0, "season_type": 3,
        "team1_moneyline": None, "team2_moneyline": None,
        "team1_spread": 0.0, "team2_spread": 0.0,
        "team1_spread_odds": None, "team2_spread_odds": None,
        "total_score": 0.0, "over_odds": None, "under_odds": None,
        "stats": json.dumps(raw_blob), "normalized_stats": json.dumps(norm_blob),
    }
    return pd.DataFrame([row])


# ==============================================================================
#  PREDICTION PIPELINE
# ==============================================================================

def _run_single_prediction(model_instance, loaded_info, team1_name, team2_name,
                            stats_cache, game_date):
    sklearn_model = loaded_info["model"]
    scaler        = loaded_info["scaler"]
    ohe           = loaded_info.get("one_hot_encoder")
    column        = loaded_info["column"]
    trained_idx   = loaded_info.get("trained_numerical_indices")
    cat_names     = loaded_info.get("categorical_feature_names")
    inc_spread    = loaded_info.get("include_market_spread", False)
    inc_total     = loaded_info.get("include_market_total", False)

    model_instance.column                     = column
    model_instance.trained_numerical_indices_  = trained_idx
    model_instance.categorical_feature_names   = cat_names
    model_instance.feature_engineering_mode    = loaded_info.get("feature_engineering_mode", "flatten")
    model_instance.include_market_spread       = inc_spread
    model_instance.include_market_total        = inc_total
    model_instance.one_hot_encoder             = ohe

    df        = build_game_dataframe(team1_name, team2_name, stats_cache, game_date, column)
    X_num, _  = model_instance._prepare_numerical_features(df)
    X_num_sel = X_num[:, trained_idx] if trained_idx is not None else X_num

    cats_to_use = cat_names if cat_names is not None else MLModel._DEFAULT_CATEGORICAL_FEATURES
    if ohe is not None and cats_to_use:
        cat_df = model_instance._extract_categorical_features(df, features_to_extract=cats_to_use)
        X_cat  = ohe.transform(cat_df)
    else:
        X_cat = np.empty((1, 0))

    X_final = np.hstack([X_num_sel, X_cat])
    if inc_spread:
        X_final = np.hstack([X_final, np.array([[0.0]])])
    if inc_total:
        X_final = np.hstack([X_final, np.array([[0.0]])])

    X_scaled = scaler.transform(X_final) if scaler else X_final

    if isinstance(sklearn_model, dict):
        win_prob = float(sklearn_model["win"].predict_proba(X_scaled)[0, 1])
        return {"team1_win_prob": win_prob, "team2_win_prob": 1 - win_prob}
    else:
        scores = sklearn_model.predict(X_scaled)[0]
        p1, p2 = float(scores[0]), float(scores[1])
        return {"team1_win_prob": 1.0 if p1 > p2 else 0.0,
                "team2_win_prob": 0.0 if p1 > p2 else 1.0,
                "pred_team1_score": p1, "pred_team2_score": p2}


def predict_matchup(model_instance, loaded_info, team1_name, team2_name,
                    stats_cache, game_date, round_name: str = None):
    """
    Bias-corrected neutral-site prediction.

    Runs the model twice:
      Forward pass  — team1 in slot 1 (the "away" position it was trained on)
      Reverse pass  — team2 in slot 1, then we flip back to team1's perspective

    Final P(team1 wins) = average of the two, cancelling positional home/away bias.

    Also applies INJURY_ADJUSTMENTS and OVERRIDES.

    Returns:
      team1_win_prob   final adjusted probability
      team2_win_prob
      fwd_team1_prob   raw forward-pass P(team1 wins) before averaging
      rev_team1_prob   raw reverse-pass P(team1 wins) before averaging
      overridden       True if result was forced by OVERRIDES
    """
    if round_name:
        forced = _check_override(team1_name, team2_name, round_name)
        if forced is not None:
            p1 = 1.0 if forced == team1_name else 0.0
            return {"team1_win_prob": p1, "team2_win_prob": 1.0 - p1,
                    "fwd_team1_prob": p1, "rev_team1_prob": p1, "overridden": True}

    fwd = _run_single_prediction(model_instance, loaded_info,
                                 team1_name, team2_name, stats_cache, game_date)
    rev = _run_single_prediction(model_instance, loaded_info,
                                 team2_name, team1_name, stats_cache, game_date)

    fwd_p1 = fwd["team1_win_prob"]
    rev_p1 = rev["team2_win_prob"]   # P(team1 wins | team1 was in slot 2)
    p1     = (fwd_p1 + rev_p1) / 2.0

    adj1 = INJURY_ADJUSTMENTS.get(team1_name, 1.0)
    adj2 = INJURY_ADJUSTMENTS.get(team2_name, 1.0)
    if adj1 != 1.0 or adj2 != 1.0:
        p1_adj = p1 * adj1
        p2_adj = (1.0 - p1) * adj2
        total  = p1_adj + p2_adj
        p1     = p1_adj / total

    return {"team1_win_prob": p1, "team2_win_prob": 1.0 - p1,
            "fwd_team1_prob": fwd_p1, "rev_team1_prob": rev_p1, "overridden": False}


# ==============================================================================
#  BRACKET CANVAS RENDERER
# ==============================================================================

def _bracket_center(round_idx: int, game_idx: int) -> int:
    """Canvas row for the center of (round_idx, game_idx) in a region."""
    return 2 ** (round_idx + 1) * (2 * game_idx + 1) - 1


def _simulate_region(region_name, matchups, model_instance, loaded_info,
                     stats_cache, game_date):
    """
    Simulate all 4 rounds of a region. Returns a list of 4 lists of game dicts.
    Each game dict contains: t1, s1, t2, s2, fwd, rev, p1, winner, wseed, ovr.
    """
    all_rounds = []
    current    = [(m["team1"], m["seed1"], m["team2"], m["seed2"]) for m in matchups]

    for round_idx in range(4):
        label      = ROUND_NAMES[round_idx]
        round_games = []
        next_pairs  = []

        for t1, s1, t2, s2 in current:
            res = predict_matchup(model_instance, loaded_info,
                                  t1, t2, stats_cache, game_date, round_name=label)
            p1      = res["team1_win_prob"]
            winner  = t1 if p1 >= 0.5 else t2
            wseed   = s1 if p1 >= 0.5 else s2
            round_games.append({
                "t1": t1, "s1": s1, "t2": t2, "s2": s2,
                "fwd": res["fwd_team1_prob"], "rev": res["rev_team1_prob"],
                "p1": p1, "winner": winner, "wseed": wseed,
                "ovr": res.get("overridden", False),
            })
            next_pairs.append((winner, wseed))

        all_rounds.append(round_games)
        if round_idx < 3:
            current = [
                (next_pairs[i][0], next_pairs[i][1],
                 next_pairs[i+1][0], next_pairs[i+1][1])
                for i in range(0, len(next_pairs), 2)
            ]

    return all_rounds


def _render_region(region_name: str, all_rounds: list) -> tuple:
    """
    Render a full region bracket to the terminal using a character canvas.

    Layout (columns, 0-indexed):
      0-27   R64 team names + dashes ending with bracket char at col 27
      28     R64 bracket chars (┐ │ ┘ ├)
      30-53  R32 winner text
      54     R32 bracket chars
      56-73  S16 winner text
      74     S16 bracket chars
      76-96  E8 winner text

    Center row for (round r, game g) = 2^(r+1) * (2g+1) - 1
    This means R64 centers are at 1,5,9,13,17,21,25,29 (one per 4-line slot).
    R32/S16/E8 centers land exactly at the midpoints of their input pairs.

    For each non-R64 matchup the single center row shows:
      (seed)✓ name  Fxx/Rxx
    where F/R are fwd/rev win probabilities expressed from the WINNER's perspective.
    """
    NLINES = 8 * _UNIT    # 32 canvas rows

    R64_NAME = 0   # team name starts here
    R64_BK   = 28  # ┐ │ ┘ ├  characters
    R32_COL  = 30  # R32 winner text
    R32_BK   = 54  # bracket chars for R32
    S16_COL  = 56  # S16 winner text
    S16_BK   = 74  # bracket chars for S16
    E8_COL   = 76  # E8 winner text
    CWIDTH   = 100

    canvas = [[' '] * CWIDTH for _ in range(NLINES)]

    def put(row, col, text):
        for i, ch in enumerate(text):
            if 0 <= row < NLINES and 0 <= col + i < CWIDTH:
                canvas[row][col + i] = ch

    def r64_name_line(team, seed, is_winner, ovr, end_col):
        """Build a padded/dashed name line that ends at end_col-1."""
        marker  = "✓" if is_winner else " "
        ovr_tag = "⚠ " if ovr else ""
        raw     = f"({seed:2}){marker}{ovr_tag}{team}"
        max_len = end_col - R64_NAME - 1      # leave 1 for the space before dashes
        if len(raw) > max_len:
            raw = raw[:max_len]
        dashes = "─" * max(0, end_col - R64_NAME - len(raw) - 1)
        return (raw + " " + dashes)[:end_col - R64_NAME]

    def r64_prob_line(fwd, rev, ovr):
        """Build the probability line shown at the center of an R64 game."""
        if ovr:
            return "  [MANUAL OVERRIDE]"
        # Always from team1's perspective — winner indicator on name lines makes
        # it clear which side won.
        return f"  F:{fwd*100:4.1f}%  R:{rev*100:4.1f}%"

    def result_line(team, seed, fwd, rev, p1, t1_wins, ovr, max_width):
        """
        Single-line result for R32/S16/E8 connector rows.
        Probabilities expressed from the WINNER's perspective:
          winner_fwd = P(winner wins | winner placed in slot 1)
          winner_rev = P(winner wins | winner placed in slot 2)
        """
        if ovr:
            txt = f"({seed:2})✓ {team:<12} [OVR]"
        else:
            # Derive winner's F and R from the stored t1 perspective
            wf = fwd       if t1_wins else (1.0 - rev)
            wr = rev       if t1_wins else (1.0 - fwd)
            txt = f"({seed:2})✓ {team:<12} F:{wf*100:4.1f}% R:{wr*100:4.1f}%"
        return txt[:max_width]

    # ── Draw R64 games ─────────────────────────────────────────────────
    for g, gm in enumerate(all_rounds[0]):
        c       = _bracket_center(0, g)
        t1_wins = gm["winner"] == gm["t1"]

        put(c - 1, R64_NAME, r64_name_line(gm["t1"], gm["s1"], t1_wins,     gm["ovr"], R64_BK))
        put(c - 1, R64_BK, "┐")

        put(c,     R64_NAME, r64_prob_line(gm["fwd"], gm["rev"], gm["ovr"]))
        put(c,     R64_BK,   "│")

        put(c + 1, R64_NAME, r64_name_line(gm["t2"], gm["s2"], not t1_wins, gm["ovr"], R64_BK))
        put(c + 1, R64_BK, "┘")

    # ── Draw R32 junctions and winner text ─────────────────────────────
    for g, gm in enumerate(all_rounds[1]):
        c       = _bracket_center(1, g)   # this equals R64_BK's ├ row
        t1_wins = gm["winner"] == gm["t1"]

        put(c, R64_BK, "├")
        for col in range(R64_BK + 1, R32_COL):
            put(c, col, "─")
        put(c, R32_COL, result_line(gm["winner"], gm["wseed"],
                                     gm["fwd"], gm["rev"], gm["p1"],
                                     t1_wins, gm["ovr"],
                                     R32_BK - R32_COL))
        put(c, R32_BK, "┐" if g % 2 == 0 else "┘")

    # Vertical connectors at R32_BK and S16 ├
    for pair in range(2):
        c_top = _bracket_center(1, pair * 2)
        c_bot = _bracket_center(1, pair * 2 + 1)
        c_s16 = _bracket_center(2, pair)
        for row in range(c_top + 1, c_s16):
            put(row, R32_BK, "│")
        put(c_s16, R32_BK, "├")
        for row in range(c_s16 + 1, c_bot):
            put(row, R32_BK, "│")

    # ── Draw S16 connector and winner text ─────────────────────────────
    for g, gm in enumerate(all_rounds[2]):
        c       = _bracket_center(2, g)
        t1_wins = gm["winner"] == gm["t1"]

        for col in range(R32_BK + 1, S16_COL):
            put(c, col, "─")
        put(c, S16_COL, result_line(gm["winner"], gm["wseed"],
                                     gm["fwd"], gm["rev"], gm["p1"],
                                     t1_wins, gm["ovr"],
                                     S16_BK - S16_COL))
        put(c, S16_BK, "┐" if g == 0 else "┘")

    # Vertical connectors at S16_BK and E8 ├
    c_top = _bracket_center(2, 0)
    c_bot = _bracket_center(2, 1)
    c_e8  = _bracket_center(3, 0)
    for row in range(c_top + 1, c_e8):
        put(row, S16_BK, "│")
    put(c_e8, S16_BK, "├")
    for row in range(c_e8 + 1, c_bot):
        put(row, S16_BK, "│")

    # ── Draw E8 connector and winner text ──────────────────────────────
    gm      = all_rounds[3][0]
    c       = _bracket_center(3, 0)
    t1_wins = gm["winner"] == gm["t1"]

    for col in range(S16_BK + 1, E8_COL):
        put(c, col, "─")
    put(c, E8_COL, result_line(gm["winner"], gm["wseed"],
                                gm["fwd"], gm["rev"], gm["p1"],
                                t1_wins, gm["ovr"],
                                CWIDTH - E8_COL))

    # ── Print ───────────────────────────────────────────────────────────
    sep = "═" * (CWIDTH + 4)
    print(f"\n{sep}")
    print(f"  {region_name.upper()} REGION")
    # Column header aligned to canvas positions
    r64w = R64_BK - R64_NAME
    r32w = R32_BK - R32_COL
    s16w = S16_BK - S16_COL
    print(f"  {'R64':<{r64w + 4}}{'R32':<{r32w + 4}}{'S16':<{s16w + 4}}E8")
    print("─" * (CWIDTH + 4))
    for row in canvas:
        line = "  " + "".join(row).rstrip()
        print(line)
    print()

    winner      = all_rounds[3][0]["winner"]
    winner_seed = all_rounds[3][0]["wseed"]
    print(f"  🏆 {region_name} Region Winner: ({winner_seed}) {winner}")

    return winner, winner_seed




# ==============================================================================
#  FINAL FOUR — chalk canvas display only
# ==============================================================================

def _simulate_and_render_final_four(region_winners, model_instance, loaded_info,
                                     stats_cache, game_date):
    """Chalk Final Four + Championship canvas printout."""
    sep = "═" * 70

    print(f"\n{sep}")
    print("  FINAL FOUR  (chalk path)")
    print(f"{'─' * 70}\n")

    ff_winners = []
    for region_a, region_b in FINAL_FOUR_PAIRINGS:
        t1, s1 = region_winners[region_a]
        t2, s2 = region_winners[region_b]
        res    = predict_matchup(model_instance, loaded_info,
                                 t1, t2, stats_cache, game_date,
                                 round_name="Final Four")
        p1     = res["team1_win_prob"]
        fwd    = res["fwd_team1_prob"]
        rev    = res["rev_team1_prob"]
        ovr    = res.get("overridden", False)
        winner = t1 if p1 >= 0.5 else t2
        wseed  = s1 if p1 >= 0.5 else s2

        print(f"  ({s1}) {t1:22s}  vs  ({s2}) {t2:22s}")
        if ovr:
            print(f"       [MANUAL OVERRIDE]")
        else:
            print(f"       Fwd: {fwd*100:5.1f}%   Rev: {rev*100:5.1f}%"
                  f"   Final: {p1*100:5.1f}% / {(1-p1)*100:5.1f}%")
        inj = [f"{n} ×{v:.2f}" for n, v in INJURY_ADJUSTMENTS.items() if n in (t1, t2)]
        if inj:
            print(f"       🩹 {', '.join(inj)}")
        print(f"       → ✅ {winner}\n")
        ff_winners.append((winner, wseed))

    print(f"\n{sep}")
    print("  NATIONAL CHAMPIONSHIP  (chalk path)")
    print(f"{'─' * 70}\n")

    t1, s1 = ff_winners[0]
    t2, s2 = ff_winners[1]
    res    = predict_matchup(model_instance, loaded_info,
                             t1, t2, stats_cache, game_date,
                             round_name="Championship")
    p1    = res["team1_win_prob"]
    fwd   = res["fwd_team1_prob"]
    rev   = res["rev_team1_prob"]
    ovr   = res.get("overridden", False)
    champ = t1 if p1 >= 0.5 else t2
    cseed = s1 if p1 >= 0.5 else s2

    print(f"  ({s1}) {t1:22s}  vs  ({s2}) {t2:22s}")
    if ovr:
        print(f"       [MANUAL OVERRIDE]")
    else:
        print(f"       Fwd: {fwd*100:5.1f}%   Rev: {rev*100:5.1f}%"
              f"   Final: {p1*100:5.1f}% / {(1-p1)*100:5.1f}%")
    print(f"       → ✅ {champ}\n")
    print(f"  🏆🏆🏆  CHALK CHAMPION: ({cseed}) {champ}  🏆🏆🏆\n")
    print(sep)


# ==============================================================================
#  FIRST FOUR
# ==============================================================================

def resolve_first_four(bracket, first_four, model_instance, loaded_info,
                       stats_cache, game_date):
    if not first_four:
        return

    sep = "═" * 70
    print(f"\n{sep}")
    print("  FIRST FOUR")
    print(f"{'─' * 70}\n")

    for game in first_four:
        t1, s1 = game["team1"], game["seed1"]
        t2, s2 = game["team2"], game["seed2"]
        ph     = game["replaces"]

        # Use known winner if provided, otherwise run the model
        if "winner" in game:
            winner = game["winner"]
            wseed  = game.get("winner_seed", s1 if winner == t1 else s2)
            print(f"  ({s1}) {t1:22s}  vs  ({s2}) {t2:22s}")
            print(f"       → ✅ {winner}  [known result]  (fills placeholder '{ph}')\n")
        else:
            res    = predict_matchup(model_instance, loaded_info,
                                     t1, t2, stats_cache, game_date,
                                     round_name="First Four")
            p1     = res["team1_win_prob"]
            fwd    = res["fwd_team1_prob"]
            rev    = res["rev_team1_prob"]
            ovr    = res.get("overridden", False)
            winner = t1 if p1 >= 0.5 else t2
            wseed  = s1 if p1 >= 0.5 else s2

            print(f"  ({s1}) {t1:22s}  vs  ({s2}) {t2:22s}")
            if ovr:
                print(f"       [MANUAL OVERRIDE]")
            else:
                print(f"       Fwd: {fwd*100:5.1f}%   Rev: {rev*100:5.1f}%"
                      f"   Final: {p1*100:5.1f}% / {(1-p1)*100:5.1f}%")
            print(f"       → ✅ {winner}  (fills placeholder '{ph}')\n")

        for matchups in bracket.values():
            for m in matchups:
                if m["team1"] == ph:
                    m["team1"], m["seed1"] = winner, wseed
                elif m["team2"] == ph:
                    m["team2"], m["seed2"] = winner, wseed

    print(sep)


# ==============================================================================
#  MONTE CARLO
# ==============================================================================

N_SIMULATIONS = 100_000

# ==============================================================================
#  LEVERAGE STRATEGY — REAL PUBLIC PICK PERCENTAGES
#
#  The leverage bracket picks based on:
#
#      leverage(team, slot) = P(team wins slot in MC)
#                             ─────────────────────────
#                             P(public picks team for slot)
#
#  Public pick percentages are sourced directly from ESPN bracket challenge data
#  and stored in PUBLIC_PICK_PCTS below.  Any team/round not covered falls back
#  to the seed-calibrated logistic model.
#
#  Name normalization: ESPN names are mapped to your BRACKET team names via
#  PUBLIC_NAME_MAP.  Add entries here whenever ESPN's name differs from yours.
#  First Four winners automatically inherit whichever placeholder they replaced.
# ==============================================================================

# Minimum field pick % — prevents leverage from blowing up for near-zero picks.
FIELD_PCT_FLOOR = 0.005

# ==============================================================================
#  ESPN NAME → BRACKET NAME MAP
#  Add an entry whenever ESPN's display name doesn't match your BRACKET exactly.
# ==============================================================================
PUBLIC_NAME_MAP = {
    "Connecticut":      "UConn",
    "N. Carolina":      "North Carolina",
    "Miami (FL)":       "Miami",
    "St. John's":       "St John's",
    "Michigan St.":     "Michigan St",
    "Iowa St.":         "Iowa St",
    "St. Mary's":       "Saint Mary's",
    "LIU Brooklyn":     "LIU",
    "N. Dak. St.":      "N Dakota St",
    "Northern Iowa":    "N Iowa",
    "California Baptist": "Cal Baptist",
    "Kennesaw St.":     "Kennesaw St",
    "Pennsylvania":     "Penn",
    "Wright St.":       "Wright St",
    "Tennessee St.":    "Tennessee St",
    "Furman":           "Furman",
    "Queens University":"Queens",
    "Hawaii":           "Hawai'i",
    # First Four combos — these resolve to whoever won the play-in game.
    # The resolver in build_field_model handles the actual substitution.
    "UMBC/HOW":         "UMBC",       # placeholder; replaced at runtime
    "TX/NCST":          "Texas",      # placeholder; replaced at runtime
    "PV/LEH":           "Prairie View",
    "MOH/SMU":          "Miami OH",
}

# ==============================================================================
#  PUBLIC PICK DATA
#  Key: ESPN display name (before normalization)
#  Value: list of 6 pick percentages, one per round:
#         [R64, R32, S16, E8, FF, Championship]
#  A value of None means the team was not listed for that round
#  (they were presumably eliminated — use 0.0 when calculating).
# ==============================================================================
PUBLIC_PICK_PCTS = {
    # ── 1-seeds ───────────────────────────────────────────────────────────────
    "Duke":              [98.45, 94.03, 78.46, 63.13, 47.53, 28.37],
    "Florida":           [97.05, 89.99, 76.30, 41.00, 13.65,  6.28],
    "Arizona":           [96.65, 91.38, 76.82, 59.10, 41.37, 21.92],
    "Michigan":          [96.55, 91.95, 80.82, 60.18, 27.99, 14.29],
    # ── 2-seeds ───────────────────────────────────────────────────────────────
    "Connecticut":       [96.88, 80.40, 49.91, 13.75,  7.19,  3.53],
    "Purdue":            [95.70, 81.16, 52.79, 15.66,  7.26,  3.08],
    "Houston":           [95.27, 85.73, 62.61, 35.77, 14.15,  5.74],
    "Iowa St.":          [93.96, 75.93, 54.31, 18.65,  5.37,  2.01],
    # ── 3-seeds ───────────────────────────────────────────────────────────────
    "Michigan St.":      [95.68, 77.44, 36.41,  9.25,  4.11,  1.42],
    "Gonzaga":           [93.95, 68.16, 29.88,  9.41,  4.55,  1.82],
    "Virginia":          [93.22, 63.82, 24.60,  7.08,  2.18,  0.74],
    "Illinois":          [92.59, 67.44, 21.85, 10.21,  3.40,  1.23],
    # ── 4-seeds ───────────────────────────────────────────────────────────────
    "Kansas":            [94.99, 42.15,  7.76,  4.06,  2.31,  1.41],
    "Arkansas":          [90.95, 58.28, 10.73,  6.38,  2.86,  1.22],
    "Alabama":           [87.84, 54.21,  9.21,  3.99,  1.20,  0.44],
    "Nebraska":          [86.57, 35.70,  6.01,  1.80,  0.59,  0.27],
    # ── 5-seeds ───────────────────────────────────────────────────────────────
    "St. John's":        [88.30, 53.96,  9.99,  5.37,  2.29,  1.05],
    "Vanderbilt":        [83.20, 55.95, 10.78,  2.78,  0.73,  0.26],
    "Wisconsin":         [82.85, 34.94,  6.41,  2.77,  1.39,  0.83],
    "Texas Tech":        [74.86, 34.88,  4.77,  1.74,  0.56,  0.21],
    # ── 6-seeds ───────────────────────────────────────────────────────────────
    "Tennessee":         [85.65, 30.19,  9.32,  1.83,  0.59,  0.23],
    "BYU":               [79.91, 24.81,  8.53,  1.49,  0.66,  0.26],
    "N. Carolina":       [67.68, 23.99,  9.01,  3.67,  0.88,  0.44],
    "Louisville":        [62.04, 14.57,  4.51,  0.64,  0.25,  0.11],
    # ── 7-seeds ───────────────────────────────────────────────────────────────
    "UCLA":              [68.30, 15.32,  6.03,  0.95,  0.45,  0.25],
    "Kentucky":          [66.78, 16.87,  7.39,  2.66,  1.43,  0.34],
    "Miami (FL)":        [61.75, 11.95,  4.54,  1.25,  0.42,  0.17],
    "St. Mary's":        [53.48,  5.96,  1.78,  0.56,  0.16,  0.06],
    # ── 8/9-seeds ─────────────────────────────────────────────────────────────
    "Ohio St.":          [60.07,  3.28,  1.71,  0.99,  0.32,  0.19],
    "Iowa":              [55.85,  4.67,  2.64,  0.46,  0.17,  0.08],
    "Georgia":           [55.08,  3.43,  1.23,  0.53,  0.22,  0.10],
    "Villanova":         [53.79,  4.50,  2.40,  0.76,  0.37,  0.15],
    "Saint Louis":       [42.43,  2.09,  0.96,  0.37,  0.14,  0.07],
    "Clemson":           [42.02,  3.12,  1.46,  0.54,  0.18,  0.08],
    "Utah St.":          [43.92,  1.67,  0.63,  0.24,  0.10,  0.05],
    "TCU":               [39.12,  1.58,  0.58,  0.25,  0.11,  0.05],
    # ── 10/11-seeds ───────────────────────────────────────────────────────────
    "Texas A&M":         [44.35,  5.68,  1.33,  0.47,  0.18,  0.08],
    "Missouri":          [35.90,  4.12,  1.09,  0.30,  0.12,  0.06],
    "UCF":               [30.81,  2.77,  0.73,  0.19,  0.08,  0.04],
    "Santa Clara":       [30.78,  3.73,  1.02,  0.27,  0.10,  0.05],
    "South Florida":     [36.95,  6.13,  1.13,  0.27,  0.10,  0.05],
    "VCU":               [30.23,  5.05,  0.76,  0.23,  0.09,  0.04],
    "Texas":             [17.49,  3.54,  0.65,  0.21,  0.09,  0.04],
    "MOH/SMU":           [11.81,  2.68,  0.66,  0.23,  0.10,  0.05],
    "TX/NCST":           [ 9.73,  1.90,  0.37,  0.13,  0.05,  0.03],
    # ── 12-seeds ──────────────────────────────────────────────────────────────
    "Akron":             [22.75,  6.52,  0.51,  0.21,  0.09,  0.04],
    "High Point":        [15.00,  2.81,  0.47,  0.19,  0.08,  0.03],
    "McNeese":           [14.78,  4.59,  0.57,  0.23,  0.09,  0.04],
    "Northern Iowa":     [11.05,  2.17,  0.40,  0.17,  0.07,  0.03],
    # ── 13-seeds ──────────────────────────────────────────────────────────────
    "Troy":              [11.40,  2.02,  0.40,  0.16,  0.06,  0.03],
    "Hofstra":           [ 9.86,  2.37,  0.42,  0.18,  0.08,  0.03],
    "Hawaii":            [ 6.95,  2.09,  0.53,  0.24,  0.12,  0.06],
    "California Baptist":[ 4.40,  1.09,  0.32,  0.14,  0.06,  0.03],
    # ── 14-seeds ──────────────────────────────────────────────────────────────
    "Pennsylvania":      [ 5.49,  1.81,  0.70,  0.43,  0.27,  0.17],
    "Wright St.":        [ 4.49,  1.24,  0.36,  0.14,  0.06,  0.03],
    "Kennesaw St.":      [ 3.99,  1.48,  0.32,  0.13,  0.06,  0.03],
    "N. Dak. St.":       [ 3.70,  1.20,  0.34,  0.13,  0.06,  0.03],
    # ── 15-seeds ──────────────────────────────────────────────────────────────
    "Tennessee St.":     [ 3.91,  1.54,  0.48,  0.17,  0.07,  0.03],
    "Idaho":             [ 2.95,  0.99,  0.45,  0.16,  0.07,  0.03],
    "Furman":            [ 2.59,  0.87,  0.40,  0.14,  0.06,  0.03],
    "Queens University": [ 2.34,  0.92,  0.44,  0.16,  0.07,  0.03],
    # ── 16-seeds / First Four ─────────────────────────────────────────────────
    "LIU Brooklyn":      [ 1.56,  0.71,  0.38,  0.22,  0.12,  0.06],
    "Howard":            [ 1.48,  0.65,  0.34,  0.19,  0.08,  0.04],
    "Siena":             [ 1.32,  0.61,  0.32,  0.19,  0.10,  0.04],
    "PV/LEH":            [ 1.30,  0.58,  0.31,  0.18,  0.07,  0.03],
    "UMBC/HOW":          [ 0.84,  0.37,  0.19,  0.11,  0.05,  0.02],
}


def _normalize_public_name(espn_name: str) -> str:
    """Map an ESPN display name to the bracket team name used in BRACKET."""
    return PUBLIC_NAME_MAP.get(espn_name, espn_name)


def _seed_field_pct(team_seed: int, opp_seed: int, round_idx: int,
                    team_mc_pct: float) -> float:
    """
    Fallback seed-based field estimate for teams not in PUBLIC_PICK_PCTS.
    Logistic model on seed advantage, blended lightly with MC sim%.
    """
    _ALPHA = [3.5, 2.8, 2.2, 1.8, 1.4, 1.1]
    _BETA  = [0.25, 0.22, 0.18, 0.15, 0.12, 0.10]
    advantage = opp_seed - team_seed
    logit     = _ALPHA[round_idx] + _BETA[round_idx] * advantage
    seed_frac = 1.0 / (1.0 + math.exp(-logit))
    blended   = 0.6 * seed_frac + 0.4 * (team_mc_pct / 100.0)
    return max(FIELD_PCT_FLOOR, min(1.0 - FIELD_PCT_FLOOR, blended))


def build_field_model(bracket, final_four_pairings, slot_wins, n_sims,
                      first_four_winners=None):
    """
    Build field pick percentages and leverage scores for every (slot_key, team).

    KEY DESIGN: PUBLIC_PICK_PCTS stores CUMULATIVE advance percentages — the
    fraction of all ESPN brackets where team X reached (and won) round R.
    MC slot_wins / n_sims is also a cumulative advance probability — the fraction
    of simulated tournaments where team X won their round-R game.

    Both are directly comparable WITHOUT normalization.  Do NOT normalize to
    sum to 1 within a slot — that would corrupt the leverage calculation by
    inflating low-% teams (e.g. an 8-seed with 3% real field pick would get
    boosted to ~50% after normalization against a 94% 1-seed, making them
    look like zero leverage when they might actually be high leverage).

    Leverage = MC_advance_pct / public_advance_pct  (both in [0,1])

    A leverage of 2.0 means your model gives this team twice the chance of
    reaching this round as the average bracket does — genuinely undervalued.

    Priority order for public_advance_pct:
      1. PUBLIC_PICK_PCTS[espn_name][round_idx]  — real ESPN data
      2. Seed-based logistic fallback             — for unlisted teams
    """
    field_pcts = {}
    leverage   = {}

    # Build a full seed lookup across all regions
    seed_of = {}
    for region_matchups in bracket.values():
        for m in region_matchups:
            seed_of[m["team1"]] = m["seed1"]
            seed_of[m["team2"]] = m["seed2"]

    bracket_to_espn = {v: k for k, v in PUBLIC_NAME_MAP.items()}

    def _get_public_pct(bracket_name: str, round_idx: int) -> float | None:
        """Return cumulative public advance % as a fraction [0,1], or None."""
        if bracket_name in PUBLIC_PICK_PCTS:
            pct = PUBLIC_PICK_PCTS[bracket_name][round_idx]
            return pct / 100.0 if pct is not None else None

        espn_name = bracket_to_espn.get(bracket_name)
        if espn_name and espn_name in PUBLIC_PICK_PCTS:
            pct = PUBLIC_PICK_PCTS[espn_name][round_idx]
            return pct / 100.0 if pct is not None else None

        # First Four winner: inherit the combo entry's data
        if first_four_winners:
            for placeholder, winner in first_four_winners.items():
                if winner == bracket_name:
                    for combo_espn, combo_bracket in PUBLIC_NAME_MAP.items():
                        if combo_bracket == bracket_name:
                            if combo_espn in PUBLIC_PICK_PCTS:
                                pct = PUBLIC_PICK_PCTS[combo_espn][round_idx]
                                return pct / 100.0 if pct is not None else None

        return None

    def _fp_for_team(team, round_idx, slot_data):
        """
        Return public advance % for this team/round.
        Uses real data if available, otherwise seed-based fallback.
        """
        real_pct = _get_public_pct(team, round_idx)
        if real_pct is not None:
            return max(FIELD_PCT_FLOOR, real_pct)

        # Seed fallback
        team_wins   = slot_data.get(team, 0)
        team_mc_pct = team_wins / n_sims * 100.0
        team_seed   = seed_of.get(team, 8)
        total       = sum(slot_data.values())
        opp_total   = total - team_wins
        exp_opp_seed = (
            sum(seed_of.get(o, 8) * c for o, c in slot_data.items() if o != team)
            / opp_total if opp_total > 0 else 8.5
        )
        return _seed_field_pct(team_seed, exp_opp_seed, round_idx, team_mc_pct)

    def _process_slot(slot_key, round_idx, slot_data):
        """Compute field_pct and leverage for every team in a slot."""
        for team, wins in slot_data.items():
            mc_pct = wins / n_sims          # fraction [0,1]
            fp     = _fp_for_team(team, round_idx, slot_data)
            field_pcts[(slot_key, team)] = fp
            leverage[(slot_key, team)]   = mc_pct / fp if fp > 0 else 0.0

    # ── Regions ──────────────────────────────────────────────────────────────
    for region_name, matchups in bracket.items():
        for round_idx in range(4):
            n_games = 8 >> round_idx
            for game_idx in range(n_games):
                slot_key  = (region_name, round_idx, game_idx)
                slot_data = slot_wins.get(slot_key, {})
                if slot_data:
                    _process_slot(slot_key, round_idx, slot_data)

    # ── Final Four ────────────────────────────────────────────────────────────
    for ff_idx, (ra, rb) in enumerate(final_four_pairings):
        slot_key  = ("FinalFour", ff_idx)
        slot_data = slot_wins.get(slot_key, {})
        if slot_data:
            _process_slot(slot_key, 4, slot_data)

    # ── Championship ──────────────────────────────────────────────────────────
    slot_key  = ("Championship", 0)
    slot_data = slot_wins.get(slot_key, {})
    if slot_data:
        _process_slot(slot_key, 5, slot_data)

    return field_pcts, leverage

    return field_pcts, leverage


def _get_win_prob_cached(t1, t2, round_name, prob_cache,
                         model_instance, loaded_info, stats_cache, game_date):
    """
    Returns P(t1 beats t2).  Checks OVERRIDES first, then the prob_cache
    (keyed by sorted team pair so each unique matchup is computed once).
    """
    forced = _check_override(t1, t2, round_name)
    if forced is not None:
        return 1.0 if forced == t1 else 0.0

    key = tuple(sorted([t1, t2]))
    if key not in prob_cache:
        res = predict_matchup(model_instance, loaded_info,
                              key[0], key[1], stats_cache, game_date,
                              round_name=round_name)
        prob_cache[key] = {
            "p":   res["team1_win_prob"],
            "fwd": res["fwd_team1_prob"],
            "rev": res["rev_team1_prob"],
            "ovr": res.get("overridden", False),
        }
    e = prob_cache[key]
    return e["p"] if t1 == key[0] else 1.0 - e["p"]


def _simulate_tournament_once(bracket, final_four_pairings, win_prob_fn, rng):
    slot_results     = {}
    region_champions = {}

    for region_name, matchups in bracket.items():
        current = [(m["team1"], m["seed1"], m["team2"], m["seed2"]) for m in matchups]
        for round_idx in range(4):
            rname   = ROUND_NAMES[round_idx]
            winners = []
            for game_idx, (t1, s1, t2, s2) in enumerate(current):
                p1 = win_prob_fn(t1, t2, rname)
                if rng.random() < p1:
                    slot_results[(region_name, round_idx, game_idx)] = t1
                    winners.append((t1, s1))
                else:
                    slot_results[(region_name, round_idx, game_idx)] = t2
                    winners.append((t2, s2))
            if round_idx < 3:
                current = [(winners[i][0], winners[i][1],
                            winners[i+1][0], winners[i+1][1])
                           for i in range(0, len(winners), 2)]
        region_champions[region_name] = winners[0]

    ff_winners = []
    for ff_idx, (ra, rb) in enumerate(final_four_pairings):
        t1, s1 = region_champions[ra]
        t2, s2 = region_champions[rb]
        p1 = win_prob_fn(t1, t2, "Final Four")
        if rng.random() < p1:
            slot_results[("FinalFour", ff_idx)] = t1
            ff_winners.append((t1, s1))
        else:
            slot_results[("FinalFour", ff_idx)] = t2
            ff_winners.append((t2, s2))

    t1, _ = ff_winners[0]
    t2, _ = ff_winners[1]
    p1 = win_prob_fn(t1, t2, "Championship")
    slot_results[("Championship", 0)] = t1 if rng.random() < p1 else t2
    return slot_results


def run_monte_carlo(bracket, final_four_pairings, model_instance, loaded_info,
                    stats_cache, game_date, n_sims=N_SIMULATIONS):
    """
    Run MC simulations. Returns slot_wins dict and prob_cache.
    slot_wins[(slot_key)][team_name] = number of simulations where that team won that slot.
    prob_cache holds fwd/rev probs for every unique matchup encountered.
    """
    from collections import defaultdict

    rng        = np.random.default_rng(42)
    prob_cache = {}

    def win_prob_fn(t1, t2, round_name):
        return _get_win_prob_cached(t1, t2, round_name, prob_cache,
                                    model_instance, loaded_info, stats_cache, game_date)

    # Pre-warm all R64 matchups into cache
    for matchups in bracket.values():
        for m in matchups:
            win_prob_fn(m["team1"], m["team2"], "Round of 64")

    print(f"  Running {n_sims:,} simulations ...")
    slot_wins = defaultdict(lambda: defaultdict(int))

    for i in range(n_sims):
        if (i + 1) % 2_500 == 0:
            print(f"    {i+1:,} / {n_sims:,} ...")
        for slot_key, team in _simulate_tournament_once(
                bracket, final_four_pairings, win_prob_fn, rng).items():
            slot_wins[slot_key][team] += 1

    print(f"  Done. {len(prob_cache)} unique matchup probs computed.\n")
    return dict(slot_wins), prob_cache



def _get_win_prob_cached(t1, t2, round_name, prob_cache,
                         model_instance, loaded_info, stats_cache, game_date):
    """
    Returns P(t1 beats t2).  Checks OVERRIDES first, then the prob_cache
    (keyed by sorted team pair so each unique matchup is computed once).
    """
    forced = _check_override(t1, t2, round_name)
    if forced is not None:
        return 1.0 if forced == t1 else 0.0

    key = tuple(sorted([t1, t2]))
    if key not in prob_cache:
        res = predict_matchup(model_instance, loaded_info,
                              key[0], key[1], stats_cache, game_date,
                              round_name=round_name)
        prob_cache[key] = {
            "p":   res["team1_win_prob"],
            "fwd": res["fwd_team1_prob"],
            "rev": res["rev_team1_prob"],
            "ovr": res.get("overridden", False),
        }
    e = prob_cache[key]
    return e["p"] if t1 == key[0] else 1.0 - e["p"]


# ==============================================================================
#  BRACKET PATH BUILDER
#
#  Four strategies — each independently chains its own winners forward:
#
#  chalk     — always picks the team with the higher head-to-head model prob
#  mc        — always picks whoever won more MC simulations (same as optimal)
#  optimal   — same as mc (alias kept for display labelling)
#  leverage  — picks the team with the highest leverage score:
#
#      leverage = P(team wins slot in MC)
#                 ─────────────────────────────────────────
#                 P(public picks team for this slot)
#
#      This maximises field-adjusted expected value — the picks that score
#      points most of the field didn't earn.  Public pick percentages come
#      from PUBLIC_PICK_PCTS; teams not listed fall back to a seed model.
#
#  Because picks differ across strategies, the teams playing in later rounds
#  can differ.  The MC slot_wins marginalizes over every possible path so
#  any team's slot% is always valid to look up.
# ==============================================================================

def _build_bracket_path(bracket, final_four_pairings, strategy,
                        slot_wins, n_sims, prob_cache,
                        model_instance, loaded_info, stats_cache, game_date,
                        field_pcts=None, leverage=None):
    """
    Walk through all 6 rounds, making picks according to `strategy`.
    Returns a dict:
      "regions": { region_name: [ round0_games, round1_games, ...] }
      "ff":      [ ff_game0, ff_game1 ]
      "champ":   champ_game

    Strategies: chalk | mc | optimal | leverage

    Each game dict contains:
      t1, s1, t2, s2      — teams playing
      fwd, rev, p1        — head-to-head model probs (t1's perspective)
      p1_sim, p2_sim      — Monte Carlo slot win %
      field_pct1/2        — estimated public pick % for each team
      leverage1/leverage2 — MC% / field_pct for each team
      pick, pick_seed     — who this bracket picks
      pick_leverage       — leverage score of the chosen pick
      slot_key            — for downstream lookups
      ovr                 — True if override forced the result
      took_upset          — True if leverage picked the lower-seed team
    """

    def _ensure_prob(t1, t2, round_name):
        forced = _check_override(t1, t2, round_name)
        if forced is not None:
            p1 = 1.0 if forced == t1 else 0.0
            return p1, p1, p1, True

        key = tuple(sorted([t1, t2]))
        if key not in prob_cache:
            res = predict_matchup(model_instance, loaded_info,
                                  key[0], key[1], stats_cache, game_date,
                                  round_name=round_name)
            prob_cache[key] = {
                "p":   res["team1_win_prob"],
                "fwd": res["fwd_team1_prob"],
                "rev": res["rev_team1_prob"],
                "ovr": res.get("overridden", False),
            }
        e      = prob_cache[key]
        p1_fin = e["p"]   if t1 == key[0] else 1.0 - e["p"]
        fwd    = e["fwd"] if t1 == key[0] else 1.0 - e["rev"]
        rev    = e["rev"] if t1 == key[0] else 1.0 - e["fwd"]
        return fwd, rev, p1_fin, e["ovr"]

    def _sim_pct(slot_key, team):
        return slot_wins.get(slot_key, {}).get(team, 0) / n_sims * 100

    def _lev(slot_key, team):
        if leverage is None:
            return 1.0
        return leverage.get((slot_key, team), 1.0)

    def _fp(slot_key, team):
        if field_pcts is None:
            return None
        return field_pcts.get((slot_key, team))

    def _walk():
        region_data      = {}
        region_champions = {}

        for region_name, matchups in bracket.items():
            current    = [(m["team1"], m["seed1"], m["team2"], m["seed2"]) for m in matchups]
            all_rounds = []

            for round_idx in range(4):
                rname       = ROUND_NAMES[round_idx]
                round_games = []
                next_teams  = []

                for game_idx, (t1, s1, t2, s2) in enumerate(current):
                    slot_key = (region_name, round_idx, game_idx)
                    fwd, rev, p1_fin, ovr = _ensure_prob(t1, t2, rname)
                    p1_sim = _sim_pct(slot_key, t1)
                    p2_sim = _sim_pct(slot_key, t2)
                    lev1   = _lev(slot_key, t1)
                    lev2   = _lev(slot_key, t2)
                    fp1    = _fp(slot_key, t1)
                    fp2    = _fp(slot_key, t2)

                    if ovr:
                        pick = t1 if p1_fin >= 0.5 else t2
                    elif strategy == "chalk":
                        pick = t1 if p1_fin >= 0.5 else t2
                    elif strategy == "leverage":
                        pick = t1 if lev1 >= lev2 else t2
                    else:
                        # mc / optimal — highest sim%
                        pick = t1 if p1_sim >= p2_sim else t2

                    pick_seed    = s1 if pick == t1 else s2
                    pick_lev     = lev1 if pick == t1 else lev2
                    # "upset" in leverage context = picked the lower-seeded (higher number) team
                    fav_by_sim   = t1 if p1_sim >= p2_sim else t2
                    took_upset   = (pick != fav_by_sim)

                    gm = {
                        "t1": t1, "s1": s1, "t2": t2, "s2": s2,
                        "fwd": fwd, "rev": rev, "p1": p1_fin,
                        "p1_sim": p1_sim, "p2_sim": p2_sim,
                        "field_pct1": fp1, "field_pct2": fp2,
                        "leverage1": lev1, "leverage2": lev2,
                        "pick": pick, "pick_seed": pick_seed,
                        "pick_leverage": pick_lev,
                        "slot_key": slot_key, "ovr": ovr,
                        "took_upset": took_upset,
                        # kept for compatibility with existing display/xlsx code
                        "upset_roi": None, "threshold": None,
                        "gap": abs(p1_sim - p2_sim),
                        "dog": t2 if p1_sim >= p2_sim else t1,
                        "dog_seed": s2 if p1_sim >= p2_sim else s1,
                    }
                    round_games.append(gm)
                    next_teams.append((pick, pick_seed))

                all_rounds.append(round_games)
                if round_idx < 3:
                    current = [(next_teams[i][0], next_teams[i][1],
                                next_teams[i+1][0], next_teams[i+1][1])
                               for i in range(0, len(next_teams), 2)]

            region_data[region_name]      = all_rounds
            region_champions[region_name] = (all_rounds[3][0]["pick"],
                                             all_rounds[3][0]["pick_seed"])

        # Final Four
        ff_games   = []
        ff_winners = []
        for ff_idx, (ra, rb) in enumerate(final_four_pairings):
            t1, s1 = region_champions[ra]
            t2, s2 = region_champions[rb]
            slot_key = ("FinalFour", ff_idx)
            fwd, rev, p1_fin, ovr = _ensure_prob(t1, t2, "Final Four")
            p1_sim = _sim_pct(slot_key, t1)
            p2_sim = _sim_pct(slot_key, t2)
            lev1   = _lev(slot_key, t1)
            lev2   = _lev(slot_key, t2)
            fp1    = _fp(slot_key, t1)
            fp2    = _fp(slot_key, t2)

            if ovr:
                pick = t1 if p1_fin >= 0.5 else t2
            elif strategy == "chalk":
                pick = t1 if p1_fin >= 0.5 else t2
            elif strategy == "leverage":
                pick = t1 if lev1 >= lev2 else t2
            else:
                pick = t1 if p1_sim >= p2_sim else t2

            pick_seed  = s1 if pick == t1 else s2
            pick_lev   = lev1 if pick == t1 else lev2
            fav_by_sim = t1 if p1_sim >= p2_sim else t2
            gm = {
                "t1": t1, "s1": s1, "t2": t2, "s2": s2,
                "fwd": fwd, "rev": rev, "p1": p1_fin,
                "p1_sim": p1_sim, "p2_sim": p2_sim,
                "field_pct1": fp1, "field_pct2": fp2,
                "leverage1": lev1, "leverage2": lev2,
                "pick": pick, "pick_seed": pick_seed,
                "pick_leverage": pick_lev,
                "slot_key": slot_key, "ovr": ovr,
                "took_upset": (pick != fav_by_sim),
                "upset_roi": None, "threshold": None,
                "gap": abs(p1_sim - p2_sim),
                "dog": t2 if p1_sim >= p2_sim else t1,
                "dog_seed": s2 if p1_sim >= p2_sim else s1,
            }
            ff_games.append(gm)
            ff_winners.append((pick, pick_seed))

        # Championship
        t1, s1 = ff_winners[0]
        t2, s2 = ff_winners[1]
        slot_key = ("Championship", 0)
        fwd, rev, p1_fin, ovr = _ensure_prob(t1, t2, "Championship")
        p1_sim = _sim_pct(slot_key, t1)
        p2_sim = _sim_pct(slot_key, t2)
        lev1   = _lev(slot_key, t1)
        lev2   = _lev(slot_key, t2)
        fp1    = _fp(slot_key, t1)
        fp2    = _fp(slot_key, t2)

        if ovr:
            pick = t1 if p1_fin >= 0.5 else t2
        elif strategy == "chalk":
            pick = t1 if p1_fin >= 0.5 else t2
        elif strategy == "leverage":
            pick = t1 if lev1 >= lev2 else t2
        else:
            pick = t1 if p1_sim >= p2_sim else t2

        pick_seed  = s1 if pick == t1 else s2
        pick_lev   = lev1 if pick == t1 else lev2
        fav_by_sim = t1 if p1_sim >= p2_sim else t2
        champ_game = {
            "t1": t1, "s1": s1, "t2": t2, "s2": s2,
            "fwd": fwd, "rev": rev, "p1": p1_fin,
            "p1_sim": p1_sim, "p2_sim": p2_sim,
            "field_pct1": fp1, "field_pct2": fp2,
            "leverage1": lev1, "leverage2": lev2,
            "pick": pick, "pick_seed": pick_seed,
            "pick_leverage": pick_lev,
            "slot_key": slot_key, "ovr": ovr,
            "took_upset": (pick != fav_by_sim),
            "upset_roi": None, "threshold": None,
            "gap": abs(p1_sim - p2_sim),
            "dog": t2 if p1_sim >= p2_sim else t1,
            "dog_seed": s2 if p1_sim >= p2_sim else s1,
        }

        return region_data, region_champions, ff_games, champ_game

    region_data, _, ff_games, champ_game = _walk()
    return {"regions": region_data, "ff": ff_games, "champ": champ_game}


# ==============================================================================
#  DISPLAY — prints one full bracket path with all prob detail
# ==============================================================================

def _display_bracket_path(label, path, slot_wins, n_sims):
    """
    Print every game in a bracket path with model probs, MC sim%, leverage scores,
    and expected points.
    """
    SEP  = "═" * 80
    DASH = "─" * 80

    def _slot_round_idx(slot_key):
        if slot_key[0] == "Championship": return 5
        if slot_key[0] == "FinalFour":    return 4
        return slot_key[1]

    all_games = []
    for rounds in path["regions"].values():
        for rnd in rounds:
            all_games.extend(rnd)
    all_games.extend(path["ff"])
    all_games.append(path["champ"])

    exp_pts = sum(
        gm["p1_sim"] / 100 * ROUND_POINTS[_slot_round_idx(gm["slot_key"])]
        if gm["pick"] == gm["t1"] else
        gm["p2_sim"] / 100 * ROUND_POINTS[_slot_round_idx(gm["slot_key"])]
        for gm in all_games
    )

    # Leverage-weighted expected score: sum of (leverage × MC_pts) per pick
    lev_exp_pts = sum(
        gm.get("pick_leverage", 1.0) *
        (gm["p1_sim"] / 100 if gm["pick"] == gm["t1"] else gm["p2_sim"] / 100) *
        ROUND_POINTS[_slot_round_idx(gm["slot_key"])]
        for gm in all_games
    )

    print(f"\n{SEP}")
    print(f"  {label}")
    print(f"  Expected points (MC):        {exp_pts:.2f}")
    if "LEVERAGE" in label:
        print(f"  Leverage-weighted exp pts:   {lev_exp_pts:.2f}  "
              f"(sum of leverage × P(correct) × round_pts)")
        print(f"  Field model: real ESPN public pick data  |  seed fallback for unlisted teams")
        print(f"  Leverage = P(MC wins slot) / P(field picks team)")
    print(f"  Fwd = model prob T1 in slot 1 | Rev = model prob T1 in slot 2 (flipped)")
    print(f"  Final = avg(Fwd,Rev)  |  Sim = MC slot win %")
    if OVERRIDES:
        print(f"  ⚠  Overrides active: {OVERRIDES}")
    if INJURY_ADJUSTMENTS:
        print(f"  🩹  Injury adjustments: {INJURY_ADJUSTMENTS}")
    print(SEP)

    def _print_game(gm):
        t1, s1     = gm["t1"],  gm["s1"]
        t2, s2     = gm["t2"],  gm["s2"]
        fwd        = gm["fwd"]
        rev        = gm["rev"]
        p1         = gm["p1"]
        p1_sim     = gm["p1_sim"]
        p2_sim     = gm["p2_sim"]
        lev1       = gm.get("leverage1", 1.0)
        lev2       = gm.get("leverage2", 1.0)
        fp1        = gm.get("field_pct1")
        fp2        = gm.get("field_pct2")
        pick       = gm["pick"]
        ovr        = gm["ovr"]
        took_upset = gm.get("took_upset", False)

        mk1 = "  ← PICK" if pick == t1 else ""
        mk2 = "  ← PICK" if pick == t2 else ""
        if took_upset and "LEVERAGE" in label:
            if pick == t1: mk1 = "  ⚡ LEV PICK"
            else:          mk2 = "  ⚡ LEV PICK"

        fp1_str = f"  field:{fp1*100:4.1f}%" if fp1 is not None else ""
        fp2_str = f"  field:{fp2*100:4.1f}%" if fp2 is not None else ""
        lev1_str = f"  lev:{lev1:.2f}" if gm.get("leverage1") is not None else ""
        lev2_str = f"  lev:{lev2:.2f}" if gm.get("leverage2") is not None else ""

        inj = [f"{n} ×{v:.2f}" for n, v in INJURY_ADJUSTMENTS.items() if n in (t1, t2)]
        inj_str = f"  🩹 {', '.join(inj)}" if inj else ""

        print(f"  ({s1:2}) {t1:<22s}  sim:{p1_sim:5.1f}%{fp1_str}{lev1_str}{mk1}")
        if ovr:
            print(f"        Fwd: ---     Rev: ---     Final: ---     ⚠ MANUAL OVERRIDE")
        else:
            print(f"        Fwd:{fwd*100:5.1f}%  Rev:{rev*100:5.1f}%"
                  f"  Final:{p1*100:5.1f}%{inj_str}")
        print(f"  ({s2:2}) {t2:<22s}  sim:{p2_sim:5.1f}%{fp2_str}{lev2_str}{mk2}")
        print()

    for region_name, all_rounds in path["regions"].items():
        print(f"\n  {'─'*76}")
        print(f"  {region_name.upper()} REGION")
        print(f"  {'─'*76}")
        for round_idx, round_games in enumerate(all_rounds):
            rname = ROUND_NAMES[round_idx]
            pts   = ROUND_POINTS[round_idx]
            print(f"\n  {rname}  [{pts} pt{'s' if pts > 1 else ''}]")
            print(f"  {DASH}")
            for gm in round_games:
                _print_game(gm)

    print(f"\n  {'─'*76}")
    print(f"  FINAL FOUR  [16 pts]")
    print(f"  {'─'*76}\n")
    for gm in path["ff"]:
        _print_game(gm)

    print(f"  {'─'*76}")
    print(f"  CHAMPIONSHIP  [32 pts]")
    print(f"  {'─'*76}\n")
    _print_game(path["champ"])

    champ     = path["champ"]["pick"]
    champ_sim = path["champ"]["p1_sim"] if champ == path["champ"]["t1"] \
                else path["champ"]["p2_sim"]
    champ_lev = path["champ"].get("pick_leverage", 1.0)
    print(f"  {'─'*76}")
    lev_note = f"  |  leverage: {champ_lev:.2f}" if path["champ"].get("pick_leverage") is not None else ""
    print(f"  🏆  {label} CHAMPION: {champ}  "
          f"(sim: {champ_sim:.1f}%{lev_note}  |  exp pts: {champ_sim/100*32:.2f})")
    print(f"  {SEP}\n")


# ==============================================================================
#  XLSX EXPORT
# ==============================================================================

XLSX_FILE = "bracket_output.xlsx"

def _export_xlsx(chalk_path, mc_path, optimal_path, leverage_path, today):
    """
    Write all bracket data to an Excel workbook.
    Sheets: Summary | Monte Carlo | Chalk | Optimal | Leverage
    """
    from openpyxl import Workbook
    from openpyxl.styles import (Font, PatternFill, Alignment, Border, Side)
    from openpyxl.utils import get_column_letter

    wb = Workbook()

    C_HEADER_BG  = "1F4E79"
    C_HEADER_FG  = "FFFFFF"
    C_REGION_BG  = "D6E4F0"
    C_REGION_FG  = "1F4E79"
    C_LEV_BG     = "E8D5F5"   # light purple — leverage pick differs from optimal
    C_UPSET_BG   = "FFF2CC"   # yellow — leverage took the lower-seed pick
    C_ALT_BG     = "F2F2F2"
    C_OVR_BG     = "F4CCCC"
    C_WHITE      = "FFFFFF"

    thin  = Side(style="thin",   color="BFBFBF")
    thick = Side(style="medium", color="9E9E9E")
    def _border(**_):
        return Border(top=thin, bottom=thin, left=thin, right=thin)
    def _hfont(bold=True, size=10, color=C_HEADER_FG):
        return Font(name="Arial", bold=bold, size=size, color=color)
    def _cfont(bold=False, size=10, color="000000", italic=False):
        return Font(name="Arial", bold=bold, size=size, color=color, italic=italic)
    def _fill(hex_color):
        return PatternFill("solid", start_color=hex_color, fgColor=hex_color)
    def _pct(v):
        return round(v / 100, 6) if v is not None else None

    # ── Flatten a bracket path into row dicts ─────────────────────────────────
    def _flatten(path, strategy_label):
        rows = []
        rn_full = ["Round of 64","Round of 32","Sweet 16",
                   "Elite 8","Final Four","Championship"]

        def _add(gm, region, round_idx, game_num):
            t1_wins = gm["pick"] == gm["t1"]
            rows.append({
                "region":        region,
                "round":         rn_full[round_idx],
                "round_pts":     ROUND_POINTS[round_idx],
                "game_num":      game_num,
                "team1":         gm["t1"],   "seed1":    gm["s1"],
                "team2":         gm["t2"],   "seed2":    gm["s2"],
                "fwd_t1":        gm["fwd"],  "fwd_t2":   1 - gm["fwd"],
                "rev_t1":        gm["rev"],  "rev_t2":   1 - gm["rev"],
                "final_t1":      gm["p1"],   "final_t2": 1 - gm["p1"],
                "sim_t1":        gm["p1_sim"] / 100,
                "sim_t2":        gm["p2_sim"] / 100,
                "sim_gap":       gm.get("gap", abs(gm["p1_sim"] - gm["p2_sim"])) / 100,
                "field_pct1":    gm.get("field_pct1"),
                "field_pct2":    gm.get("field_pct2"),
                "leverage1":     gm.get("leverage1"),
                "leverage2":     gm.get("leverage2"),
                "pick":          gm["pick"],
                "pick_seed":     gm["pick_seed"],
                "pick_is_t1":    t1_wins,
                "pick_sim":      (gm["p1_sim"] if t1_wins else gm["p2_sim"]) / 100,
                "pick_field_pct": gm.get("field_pct1") if t1_wins else gm.get("field_pct2"),
                "pick_leverage": gm.get("pick_leverage"),
                "took_upset":    gm.get("took_upset", False),
                "override":      gm["ovr"],
                "strategy":      strategy_label,
                # compatibility stubs
                "threshold": None, "upset_roi": None, "is_candidate": False,
            })

        for region_name, all_rounds in path["regions"].items():
            for round_idx, round_games in enumerate(all_rounds):
                for game_num, gm in enumerate(round_games, 1):
                    _add(gm, region_name, round_idx, game_num)
        for i, gm in enumerate(path["ff"], 1):
            _add(gm, "Final Four", 4, i)
        _add(path["champ"], "Championship", 5, 1)
        return rows

    chalk_rows    = _flatten(chalk_path,    "Chalk")
    mc_rows       = _flatten(mc_path,       "Monte Carlo")
    optimal_rows  = _flatten(optimal_path,  "Optimal")
    leverage_rows = _flatten(leverage_path, "Leverage")

    def _exp_pts(rows):
        return sum(r["pick_sim"] * r["round_pts"] for r in rows)

    # ── Detail sheet ──────────────────────────────────────────────────────────
    def _write_detail_sheet(ws, rows, strategy_label, include_leverage=False):
        ws.sheet_view.showGridLines = False
        exp_pts = _exp_pts(rows)

        ws.append([f"{strategy_label}  —  {today.strftime('%B %d, %Y')}"])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=24)
        ws["A1"].font      = Font(name="Arial", bold=True, size=13, color=C_HEADER_FG)
        ws["A1"].fill      = _fill(C_HEADER_BG)
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

        subtitle = f"Expected Score: {exp_pts:.2f} pts  (sum of P(pick wins slot) × round pts)"
        ws.append([subtitle])
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=24)
        ws["A2"].font      = Font(name="Arial", bold=True, size=11, color=C_HEADER_FG)
        ws["A2"].fill      = _fill("2E6A9E")
        ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 18

        base_cols = [
            "Region", "Round", "Pts",
            "Team 1", "Seed 1", "Team 2", "Seed 2",
            "Fwd T1%", "Fwd T2%", "Rev T1%", "Rev T2%",
            "Final T1%", "Final T2%",
            "Sim T1%", "Sim T2%", "Sim Gap%",
            "Pick", "Seed", "Pick Sim%", "Override",
        ]
        lev_cols = [
            "Field % T1", "Field % T2",
            "Leverage T1", "Leverage T2", "Pick Leverage", "Lev Pick?",
        ]
        headers = base_cols + (lev_cols if include_leverage else [])

        ws.append(headers)
        hrow = ws.max_row
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=hrow, column=ci, value=h)
            c.font = _hfont(); c.fill = _fill(C_HEADER_BG)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = _border()
        ws.row_dimensions[hrow].height = 28

        prev_region = prev_round = None
        alt = False
        for r in rows:
            if r["region"] != prev_region or r["round"] != prev_round:
                sep_val = f"  {r['region']}  ·  {r['round']}"
                ws.append([sep_val] + [""] * (len(headers) - 1))
                sr = ws.max_row
                ws.merge_cells(start_row=sr, start_column=1, end_row=sr, end_column=len(headers))
                c = ws.cell(row=sr, column=1)
                c.font = Font(name="Arial", bold=True, size=10, color=C_REGION_FG)
                c.fill = _fill(C_REGION_BG)
                c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
                ws.row_dimensions[sr].height = 18
                prev_region = r["region"]; prev_round = r["round"]; alt = False

            if r["override"]:       bg = C_OVR_BG
            elif r["took_upset"] and include_leverage: bg = C_UPSET_BG
            elif alt:               bg = C_ALT_BG
            else:                   bg = C_WHITE
            alt = not alt

            row_vals = [
                r["region"], r["round"], r["round_pts"],
                r["team1"], r["seed1"], r["team2"], r["seed2"],
                _pct(r["fwd_t1"]*100), _pct(r["fwd_t2"]*100),
                _pct(r["rev_t1"]*100), _pct(r["rev_t2"]*100),
                _pct(r["final_t1"]*100), _pct(r["final_t2"]*100),
                r["sim_t1"], r["sim_t2"], r["sim_gap"],
                r["pick"], r["pick_seed"], r["pick_sim"],
                "YES" if r["override"] else "",
            ]
            if include_leverage:
                row_vals += [
                    r["field_pct1"], r["field_pct2"],
                    round(r["leverage1"], 3) if r["leverage1"] is not None else "",
                    round(r["leverage2"], 3) if r["leverage2"] is not None else "",
                    round(r["pick_leverage"], 3) if r["pick_leverage"] is not None else "",
                    "YES" if r["took_upset"] else "",
                ]

            ws.append(row_vals)
            dr = ws.max_row
            pick_col = headers.index("Pick") + 1
            for ci in range(1, len(row_vals) + 1):
                cell = ws.cell(row=dr, column=ci)
                cell.fill      = _fill(bg)
                cell.font      = _cfont(bold=(ci == pick_col),
                                        color=("C00000" if r["override"] else "000000"))
                cell.border    = _border()
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if ci in (8,9,10,11,12,13,14,15,16,19):
                    cell.number_format = "0.0%"
                if include_leverage and ci in (21, 22):
                    cell.number_format = "0.0%"

        widths = [13,14,4,20,5,20,5, 8,8,8,8,9,9, 8,8,8, 20,5,9,8]
        if include_leverage:
            widths += [10, 10, 10, 10, 10, 9]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = ws.cell(row=3, column=1)

    # ── MC sheet ──────────────────────────────────────────────────────────────
    def _write_mc_sheet(ws, rows):
        ws.sheet_view.showGridLines = False

        # Expected score = sum of max(sim_t1, sim_t2) * round_pts for each game
        # (the MC "pick" is always whoever has the higher sim%)
        exp_pts_mc = sum(max(r["sim_t1"], r["sim_t2"]) * r["round_pts"] for r in rows)

        ws.append([f"Monte Carlo Raw Results  —  {today.strftime('%B %d, %Y')}  |  {N_SIMULATIONS:,} simulations"])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=13)
        ws["A1"].font = Font(name="Arial", bold=True, size=13, color=C_HEADER_FG)
        ws["A1"].fill = _fill(C_HEADER_BG)
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

        subtitle = (f"Expected Score: {exp_pts_mc:.2f} pts  "
                    f"(always picks higher sim% team — identical to Optimal bracket)")
        ws.append([subtitle])
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=13)
        ws["A2"].font      = Font(name="Arial", bold=True, size=11, color=C_HEADER_FG)
        ws["A2"].fill      = _fill("2E6A9E")
        ws["A2"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 18

        headers = ["Region", "Round", "Pts",
                   "Team 1", "Seed 1", "T1 Slot Sim%",
                   "Team 2", "Seed 2", "T2 Slot Sim%",
                   "Sim Pick", "Pick Seed", "Pick Sim%", "Sim Gap%"]
        ws.append(headers)
        hrow = ws.max_row
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=hrow, column=ci, value=h)
            c.font = _hfont(); c.fill = _fill(C_HEADER_BG)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = _border()
        ws.row_dimensions[hrow].height = 28

        prev_region = prev_round = None; alt = False
        for r in rows:
            if r["region"] != prev_region or r["round"] != prev_round:
                sep = f"  {r['region']}  ·  {r['round']}"
                ws.append([sep] + [""] * (len(headers) - 1))
                sr = ws.max_row
                ws.merge_cells(start_row=sr, start_column=1, end_row=sr, end_column=len(headers))
                c = ws.cell(row=sr, column=1)
                c.font = Font(name="Arial", bold=True, size=10, color=C_REGION_FG)
                c.fill = _fill(C_REGION_BG)
                c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
                ws.row_dimensions[sr].height = 18
                prev_region = r["region"]; prev_round = r["round"]; alt = False
            bg = C_ALT_BG if alt else C_WHITE; alt = not alt
            t1_wins   = r["sim_t1"] >= r["sim_t2"]
            sim_pick  = r["team1"] if t1_wins else r["team2"]
            pick_seed = r["seed1"] if t1_wins else r["seed2"]
            pick_sim  = r["sim_t1"] if t1_wins else r["sim_t2"]
            sim_gap   = abs(r["sim_t1"] - r["sim_t2"])
            row_vals = [r["region"], r["round"], r["round_pts"],
                        r["team1"], r["seed1"], r["sim_t1"],
                        r["team2"], r["seed2"], r["sim_t2"],
                        sim_pick, pick_seed, pick_sim, sim_gap]
            ws.append(row_vals)
            dr = ws.max_row
            for ci in range(1, len(row_vals)+1):
                c = ws.cell(row=dr, column=ci)
                c.fill = _fill(bg); c.border = _border()
                c.font = _cfont(bold=(ci == 10))
                c.alignment = Alignment(horizontal="center", vertical="center")
                if ci in (6, 9, 12, 13): c.number_format = "0.0%"
        for i, w in enumerate([13,14,4,20,5,9,20,5,9,20,5,9,9], 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = ws.cell(row=3, column=1)

    # ── Summary sheet ─────────────────────────────────────────────────────────
    def _write_summary(ws):
        ws.sheet_view.showGridLines = False
        total_cols = 21
        title = (f"Bracket Summary  —  {today.strftime('%B %d, %Y')}  |  "
                 f"Strategies: Chalk | MC | Optimal | Leverage  |  "
                 f"Field: real ESPN pick data")
        ws.append([title])
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=total_cols)
        ws["A1"].font      = Font(name="Arial", bold=True, size=13, color=C_HEADER_FG)
        ws["A1"].fill      = _fill(C_HEADER_BG)
        ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 22

        headers = [
            "Region", "Round", "Pts",
            "Chalk T1", "Seed 1", "Chalk T2", "Seed 2",
            "H2H Final T1%", "MC T1 Sim%", "MC T2 Sim%",
            "Chalk Pick", "MC Pick", "Optimal Pick", "Leverage Pick",
            "Opt≠Chalk", "Lev≠Opt", "Lev Took Upset",
            "Lev T1 Field%", "Lev T2 Field%",
            "Lev T1 Lev", "Override",
        ]
        ws.append(headers)
        hrow = ws.max_row
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=hrow, column=ci, value=h)
            c.font = _hfont(); c.fill = _fill(C_HEADER_BG)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = _border()
        ws.row_dimensions[hrow].height = 30

        prev_region = prev_round = None; alt = False
        pick_cols = {11, 12, 13, 14}   # Chalk, MC, Optimal, Leverage — all bolded
        for ck, mc, op, lv in zip(chalk_rows, mc_rows, optimal_rows, leverage_rows):
            if ck["region"] != prev_region or ck["round"] != prev_round:
                sep = f"  {ck['region']}  ·  {ck['round']}"
                ws.append([sep] + [""] * (len(headers) - 1))
                sr = ws.max_row
                ws.merge_cells(start_row=sr, start_column=1, end_row=sr, end_column=len(headers))
                c = ws.cell(row=sr, column=1)
                c.font = Font(name="Arial", bold=True, size=10, color=C_REGION_FG)
                c.fill = _fill(C_REGION_BG)
                c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
                ws.row_dimensions[sr].height = 18
                prev_region = ck["region"]; prev_round = ck["round"]; alt = False

            # MC pick = whoever had higher sim% (same logic as mc sheet)
            mc_pick = mc["team1"] if mc["sim_t1"] >= mc["sim_t2"] else mc["team2"]

            opt_diff = op["pick"] != ck["pick"]
            lev_diff = lv["pick"] != op["pick"]
            lev_upst = lv["took_upset"]
            ovr      = ck["override"]

            if ovr:          bg = C_OVR_BG
            elif lev_upst:   bg = C_UPSET_BG
            elif lev_diff:   bg = C_LEV_BG
            elif opt_diff:   bg = "E2EFDA"
            elif alt:        bg = C_ALT_BG
            else:            bg = C_WHITE
            alt = not alt

            fp1  = lv.get("field_pct1")
            fp2  = lv.get("field_pct2")
            lev1 = lv.get("leverage1")

            row_vals = [
                ck["region"], ck["round"], ck["round_pts"],
                ck["team1"], ck["seed1"], ck["team2"], ck["seed2"],
                ck["final_t1"], mc["sim_t1"], mc["sim_t2"],
                ck["pick"], mc_pick, op["pick"], lv["pick"],
                "YES" if opt_diff else "",
                "YES" if lev_diff else "",
                "YES" if lev_upst else "",
                fp1, fp2,
                round(lev1, 2) if lev1 is not None else "",
                "YES" if ovr else "",
            ]
            ws.append(row_vals)
            dr = ws.max_row
            for ci in range(1, len(row_vals)+1):
                c = ws.cell(row=dr, column=ci)
                c.fill = _fill(bg); c.border = _border()
                c.font = _cfont(bold=(ci in pick_cols))
                c.alignment = Alignment(horizontal="center", vertical="center")
                if ci in (8, 9, 10): c.number_format = "0.0%"
                if ci in (18, 19):   c.number_format = "0.0%"

        widths = [13,14,4,20,5,20,5, 10,9,9, 20,20,20,20, 10,9,14, 12,12,8,9]
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.freeze_panes = ws.cell(row=3, column=1)

    # ── Build workbook ────────────────────────────────────────────────────────
    ws_summary = wb.active
    ws_summary.title = "Summary"
    _write_summary(ws_summary)

    ws_mc = wb.create_sheet("Monte Carlo")
    _write_mc_sheet(ws_mc, mc_rows)

    ws_chalk = wb.create_sheet("Chalk")
    _write_detail_sheet(ws_chalk, chalk_rows, "Chalk Bracket")

    ws_opt = wb.create_sheet("Optimal")
    _write_detail_sheet(ws_opt, optimal_rows, "Optimal Bracket (MC)")

    ws_lev = wb.create_sheet("Leverage")
    _write_detail_sheet(ws_lev, leverage_rows,
                        f"Leverage Bracket  |  field: real ESPN pick data",
                        include_leverage=True)

    wb.save(XLSX_FILE)
    return XLSX_FILE


# ==============================================================================
#  OUTPUT FILE  (text mirror of terminal)
# ==============================================================================

OUTPUT_FILE = "bracket_output.txt"


class _Tee:
    """Mirror every print() to both the terminal and OUTPUT_FILE."""
    def __init__(self, filepath):
        self._file   = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)
    def flush(self):
        self._file.flush()
        self._stdout.flush()
    def close(self):
        self._file.close()


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    tee = _Tee(OUTPUT_FILE)
    sys.stdout = tee
    try:
        _run()
    finally:
        sys.stdout = tee._stdout
        tee.close()
        print(f"\nOutput saved to: {OUTPUT_FILE}")


def _run():
    today = date.today()
    sep   = "=" * 70
    print(sep)
    print(f"  MARCH MADNESS BRACKET SIMULATOR")
    print(f"  Stats as of: {today.strftime('%B %d, %Y')}")
    print(f"  Strategies: Chalk | Optimal | Leverage")
    print(f"  Field model: real ESPN pick data  |  floor: {FIELD_PCT_FLOOR:.1%}")
    if OVERRIDES:
        print(f"  ⚠️  Manual overrides: {list(OVERRIDES.keys())}")
    if INJURY_ADJUSTMENTS:
        print(f"  🩹  Injury adjustments: {list(INJURY_ADJUSTMENTS.keys())}")
    print(sep)

    # 1. Load model
    print(f"\n[1/5] Loading model: {MODEL_NAME} ...")
    model_instance = MLModel(MODEL_NAME)
    try:
        loaded_info = model_instance._load_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print("      Model loaded.\n")

    # 2. Fetch stats
    placeholders = {g["replaces"] for g in FIRST_FOUR}
    all_teams    = set()
    for region_matchups in BRACKET.values():
        for m in region_matchups:
            if m["team1"] not in placeholders: all_teams.add(m["team1"])
            if m["team2"] not in placeholders: all_teams.add(m["team2"])
    for g in FIRST_FOUR:
        all_teams.add(g["team1"])
        all_teams.add(g["team2"])

    print(f"[2/5] Fetching stats for {len(all_teams)} teams ...")
    pg          = Pregame(today, "CBB")
    stats_cache = {}
    failed      = []
    for team in sorted(all_teams):
        stats = pg.get_team_stats(team)
        if stats is None:
            print(f"  ⚠️  No stats for '{team}'")
            failed.append(team)
        else:
            stats_cache[team] = stats
            print(f"  ✅ {team}")
    if failed:
        print(f"\n  Could not fetch: {failed}\n  Fix team names and re-run.")
        sys.exit(1)
    print("\n  All stats fetched.\n")

    # 2.5 First Four — resolve play-in games and track winners for field model
    first_four_winners = {}  # {"A": "UMBC", "B": "Lehigh", ...}
    resolve_first_four(BRACKET, FIRST_FOUR, model_instance, loaded_info, stats_cache, today)
    # After resolve_first_four the BRACKET is updated in-place; reconstruct the winner map
    for g in FIRST_FOUR:
        ph = g["replaces"]
        for matchups in BRACKET.values():
            for m in matchups:
                if m.get("team1") not in (g["team1"], g["team2"]) and \
                   m.get("team2") not in (g["team1"], g["team2"]):
                    continue
                winner = m["team1"] if m["team1"] in (g["team1"], g["team2"]) else m["team2"]
                first_four_winners[ph] = winner

    # 3. Chalk bracket — visual canvas
    print(f"\n[3/5] Rendering chalk bracket canvas ...")
    region_winners = {}
    for region_name, matchups in BRACKET.items():
        rounds = _simulate_region(region_name, matchups, model_instance,
                                  loaded_info, stats_cache, today)
        winner, seed = _render_region(region_name, rounds)
        region_winners[region_name] = (winner, seed)
    _simulate_and_render_final_four(region_winners, model_instance,
                                    loaded_info, stats_cache, today)

    # 4. Monte Carlo
    print(f"\n[4/5] Running Monte Carlo ({N_SIMULATIONS:,} sims) ...")
    slot_wins, prob_cache = run_monte_carlo(
        BRACKET, FINAL_FOUR_PAIRINGS,
        model_instance, loaded_info, stats_cache, today,
        n_sims=N_SIMULATIONS,
    )

    # 5. Build field model and all bracket paths
    print(f"\n[5/5] Building field model and bracket paths ...")
    print(f"  Field model: real ESPN public pick data  |  "
          f"seed fallback for unlisted teams  |  floor={FIELD_PCT_FLOOR:.1%}")
    field_pcts, leverage = build_field_model(
        BRACKET, FINAL_FOUR_PAIRINGS, slot_wins, N_SIMULATIONS,
        first_four_winners=first_four_winners)
    print(f"  Field model built: {len(field_pcts)} (slot, team) pairs.\n")

    kwargs = dict(
        bracket             = BRACKET,
        final_four_pairings = FINAL_FOUR_PAIRINGS,
        slot_wins           = slot_wins,
        n_sims              = N_SIMULATIONS,
        prob_cache          = prob_cache,
        model_instance      = model_instance,
        loaded_info         = loaded_info,
        stats_cache         = stats_cache,
        game_date           = today,
        field_pcts          = field_pcts,
        leverage            = leverage,
    )

    chalk_path    = _build_bracket_path(strategy="chalk",    **kwargs)
    mc_path       = _build_bracket_path(strategy="mc",       **kwargs)
    optimal_path  = _build_bracket_path(strategy="optimal",  **kwargs)
    leverage_path = _build_bracket_path(strategy="leverage", **kwargs)

    _display_bracket_path("CHALK BRACKET",    chalk_path,    slot_wins, N_SIMULATIONS)
    _display_bracket_path("OPTIMAL BRACKET",  optimal_path,  slot_wins, N_SIMULATIONS)
    _display_bracket_path("LEVERAGE BRACKET", leverage_path, slot_wins, N_SIMULATIONS)

    # 6. Export to Excel
    print(f"\nExporting to Excel ...")
    xlsx = _export_xlsx(chalk_path, mc_path, optimal_path, leverage_path, today)
    print(f"Saved: {xlsx}")


if __name__ == "__main__":
    main()