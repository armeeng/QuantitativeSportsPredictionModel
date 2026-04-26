#!/usr/bin/env python3
"""
March Madness Brier Score Calculator
=====================================
Reads the bracket structure and predicted probabilities from bracket_simulator.py,
then interactively asks you for each actual game result.

Brier Score = (1/N) * sum( (p - o)^2 )
  p = model's predicted probability for team1 winning
  o = 1 if team1 actually won, 0 if team2 won
  Lower is better. A coin-flip baseline scores 0.25.

HOW TO RUN:
  python brier_score.py

REQUIREMENTS:
  - bracket_simulator.py must be in the same directory (or on sys.path)
  - All the same dependencies as bracket_simulator.py (model, Pregame, etc.)

The script will prompt you for each game's actual winner.
Type the winner's name (or a unique prefix) and press Enter.
Type 'skip' to skip a game (e.g. it hasn't been played yet).
Type 'quit' to stop early and see partial results.
"""

import os
import sys
import re
from datetime import date

# ── Allow bracket_simulator to be found in the same directory ─────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ── Import everything we need from the simulator ──────────────────────────────
from bracket_simulator import (
    BRACKET,
    FIRST_FOUR,
    FINAL_FOUR_PAIRINGS,
    ROUND_NAMES,
    MODEL_NAME,
    predict_matchup,
    resolve_first_four,
    Pregame,
)
from Model import MLModel


# ==============================================================================
#  HELPERS
# ==============================================================================

def fuzzy_match(user_input: str, candidates: list[str]) -> str | None:
    """
    Return the matching candidate if the user's input uniquely identifies one.
    Tries:
      1. Exact match (case-insensitive)
      2. The candidate starts with the input (case-insensitive)
      3. The input is a substring of the candidate (case-insensitive)
    Returns None if 0 or >1 candidates match.
    """
    u = user_input.strip().lower()
    # 1. exact
    for c in candidates:
        if c.lower() == u:
            return c
    # 2. prefix
    prefix_matches = [c for c in candidates if c.lower().startswith(u)]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return None  # ambiguous
    # 3. substring
    sub_matches = [c for c in candidates if u in c.lower()]
    if len(sub_matches) == 1:
        return sub_matches[0]
    return None


def ask_winner(team1: str, team2: str, p1: float, round_name: str) -> str | None:
    """
    Prompt the user to enter the actual winner of a game.
    Returns the winning team name, or None if skipped.
    """
    p2 = 1.0 - p1
    print(f"\n  {'─'*60}")
    print(f"  [{round_name}]")
    print(f"  (1) {team1:<25}  model: {p1*100:5.1f}%")
    print(f"  (2) {team2:<25}  model: {p2*100:5.1f}%")

    while True:
        raw = input("  Who actually won? (name/prefix, '1', '2', skip, quit): ").strip()
        if not raw:
            continue
        low = raw.lower()
        if low == "quit":
            return "QUIT"
        if low == "skip":
            return None
        if raw == "1":
            return team1
        if raw == "2":
            return team2
        match = fuzzy_match(raw, [team1, team2])
        if match:
            return match
        print(f"  ⚠  Couldn't match '{raw}' to '{team1}' or '{team2}'. Try again.")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    today = date(2026, 3, 17) #date.today()
    sep   = "=" * 64

    print(sep)
    print("  MARCH MADNESS BRIER SCORE CALCULATOR")
    print(f"  {today.strftime('%B %d, %Y')}")
    print(sep)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print(f"\n[1/3] Loading model: {MODEL_NAME} ...")
    model_instance = MLModel(MODEL_NAME)
    try:
        loaded_info = model_instance._load_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print("      Model loaded.\n")

    # ── 2. Fetch stats ────────────────────────────────────────────────────────
    placeholders = {g["replaces"] for g in FIRST_FOUR}
    all_teams    = set()
    for region_matchups in BRACKET.values():
        for m in region_matchups:
            if m["team1"] not in placeholders: all_teams.add(m["team1"])
            if m["team2"] not in placeholders: all_teams.add(m["team2"])
    for g in FIRST_FOUR:
        all_teams.add(g["team1"])
        all_teams.add(g["team2"])

    print(f"[2/3] Fetching stats for {len(all_teams)} teams ...")
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
        print(f"\n  Could not fetch stats for: {failed}")
        sys.exit(1)
    print("\n  All stats fetched.\n")

    # Resolve First Four play-ins (updates BRACKET in-place)
    resolve_first_four(BRACKET, FIRST_FOUR, model_instance, loaded_info, stats_cache, today)

    # ── 3. Walk the bracket interactively ─────────────────────────────────────
    print(f"\n[3/3] Enter actual results game by game.\n")
    print("      The model's predicted probability for each team is shown.")
    print("      Brier Score = avg (p - outcome)^2  |  lower = better")
    print("      Coin-flip baseline: 0.2500\n")

    brier_terms = []   # list of (team1, team2, p1, outcome, round_name)
    region_winners = {}

    def get_prob(t1, t2, rname):
        res = predict_matchup(model_instance, loaded_info,
                              t1, t2, stats_cache, today, round_name=rname)
        return res["team1_win_prob"]

    # ── Regions (R64 → R32 → S16 → E8) ───────────────────────────────────────
    for region_name, matchups in BRACKET.items():
        print(f"\n{'═'*64}")
        print(f"  {region_name.upper()} REGION")
        print(f"{'═'*64}")

        current = [(m["team1"], m["seed1"], m["team2"], m["seed2"]) for m in matchups]
        quit_flag = False

        for round_idx in range(4):
            rname    = ROUND_NAMES[round_idx]
            winners  = []

            for t1, s1, t2, s2 in current:
                p1     = get_prob(t1, t2, rname)
                result = ask_winner(t1, t2, p1, rname)

                if result == "QUIT":
                    quit_flag = True
                    break

                if result is None:
                    # Skipped — use model's chalk pick so bracket can continue
                    chalk = t1 if p1 >= 0.5 else t2
                    chalk_seed = s1 if p1 >= 0.5 else s2
                    print(f"  ↳ Skipped. Using chalk pick ({chalk}) to advance.")
                    winners.append((chalk, chalk_seed))
                    continue

                outcome = 1 if result == t1 else 0
                brier_terms.append((t1, t2, p1, outcome, rname, region_name))
                winner_seed = s1 if result == t1 else s2
                winners.append((result, winner_seed))
                print(f"  ✓ Recorded: {result} won  |  error² = {(p1 - outcome)**2:.4f}")

            if quit_flag:
                break

            if round_idx < 3:
                current = [
                    (winners[i][0], winners[i][1],
                     winners[i+1][0], winners[i+1][1])
                    for i in range(0, len(winners), 2)
                ]

        if quit_flag:
            break

        region_winners[region_name] = winners[0]  # (team, seed)

    # ── Final Four ────────────────────────────────────────────────────────────
    if not quit_flag and len(region_winners) == 4:
        print(f"\n{'═'*64}")
        print("  FINAL FOUR")
        print(f"{'═'*64}")

        ff_winners = []
        for region_a, region_b in FINAL_FOUR_PAIRINGS:
            t1, s1 = region_winners[region_a]
            t2, s2 = region_winners[region_b]
            p1     = get_prob(t1, t2, "Final Four")
            result = ask_winner(t1, t2, p1, "Final Four")

            if result == "QUIT":
                quit_flag = True
                break
            if result is None:
                chalk = t1 if p1 >= 0.5 else t2
                chalk_seed = s1 if p1 >= 0.5 else s2
                ff_winners.append((chalk, chalk_seed))
                continue

            outcome = 1 if result == t1 else 0
            brier_terms.append((t1, t2, p1, outcome, "Final Four", "Final Four"))
            w_seed = s1 if result == t1 else s2
            ff_winners.append((result, w_seed))
            print(f"  ✓ Recorded: {result} won  |  error² = {(p1 - outcome)**2:.4f}")

        # ── Championship ──────────────────────────────────────────────────────
        if not quit_flag and len(ff_winners) == 2:
            print(f"\n{'═'*64}")
            print("  NATIONAL CHAMPIONSHIP")
            print(f"{'═'*64}")

            t1, s1 = ff_winners[0]
            t2, s2 = ff_winners[1]
            p1     = get_prob(t1, t2, "Championship")
            result = ask_winner(t1, t2, p1, "Championship")

            if result not in (None, "QUIT"):
                outcome = 1 if result == t1 else 0
                brier_terms.append((t1, t2, p1, outcome, "Championship", "Championship"))
                print(f"  ✓ Recorded: {result} won  |  error² = {(p1 - outcome)**2:.4f}")

    # ── Print Results ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*64}")
    print("  BRIER SCORE RESULTS")
    print(f"{'='*64}\n")

    if not brier_terms:
        print("  No games recorded.")
        return

    total_sq_err = sum((p - o) ** 2 for _, _, p, o, _, _ in brier_terms)
    n            = len(brier_terms)
    brier        = total_sq_err / n

    # Per-round breakdown
    from collections import defaultdict
    round_errors = defaultdict(list)
    for t1, t2, p, o, rname, region in brier_terms:
        round_errors[rname].append((p - o) ** 2)

    print(f"  {'Round':<20} {'Games':>5}  {'Brier':>7}")
    print(f"  {'─'*36}")
    for rname in ROUND_NAMES + ["Final Four", "Championship"]:
        if rname in round_errors:
            errs = round_errors[rname]
            rb   = sum(errs) / len(errs)
            print(f"  {rname:<20} {len(errs):>5}  {rb:>7.4f}")

    print(f"  {'─'*36}")
    print(f"  {'OVERALL':<20} {n:>5}  {brier:>7.4f}")
    print(f"\n  Coin-flip baseline (0.5 every game): 0.2500")
    delta = brier - 0.25
    if delta < 0:
        print(f"  Your model beat the coin flip by   {abs(delta):.4f}  ✅")
    else:
        print(f"  Your model trailed the coin flip by {delta:.4f}  ❌")

    # Full game log
    print(f"\n\n  {'─'*64}")
    print(f"  FULL GAME LOG")
    print(f"  {'─'*64}")
    print(f"  {'Round':<16} {'Team 1':<18} {'p1':>6}  {'Team 2':<18} {'Winner':<18} {'Err²':>6}")
    print(f"  {'─'*90}")
    for t1, t2, p, o, rname, region in brier_terms:
        winner = t1 if o == 1 else t2
        print(f"  {rname:<16} {t1:<18} {p*100:5.1f}%  {t2:<18} {winner:<18} {(p-o)**2:.4f}")

    print(f"\n  Final Brier Score: {brier:.4f}\n")


if __name__ == "__main__":
    main()