#!/usr/bin/env python3
import re
from Model import MLModel
from TestModel import TestModel

def build_model_name(model_type: str, column: str, query: str) -> str:
    # 1) model abbreviation
    abbrs = {
        'linear_regression': 'lr',
        'random_forest':     'rf',
        'xgboost':           'xgb',
        'neural_network':    'nn'
        #'mlp':               'nn',
        #'mlp_regressor':     'nn',
    }
    m_abbr = abbrs.get(model_type, model_type[:2])

    # 2) column part
    col_part = 'norm' if column == 'normalized_stats' else 'nonorm'

    # 3) sport
    m = re.search(r"sport\s*=\s*'([^']+)'", query, re.IGNORECASE)
    sport = m.group(1).upper() if m else 'ALL'

    # 4) suffix: “all” if only sport is in the WHERE clause, else slug of the rest
    where = query.split('WHERE',1)[1]
    # normalize whitespace & strip trailing ;
    cond = where.strip().rstrip(';').strip()
    # check if it’s exactly just the sport clause
    if re.fullmatch(rf"sport\s*=\s*'{sport}'", cond, re.IGNORECASE):
        suffix = 'all'
    else:
        # remove sport=... then slugify remaining
        rest = re.sub(rf"sport\s*=\s*'{sport}'\s*(AND\s*)?", '', cond, flags=re.IGNORECASE)
        # slugify: letters+digits and other things only, underscores
        suffix = re.sub(r'[^0-9A-Za-z<>=]+', '_', rest).strip('_').lower()

    return f"{m_abbr}_{col_part}_{sport}_{suffix}"

def main():

    features = [
        # Win% (all games)
        'team1_stats.win-pct-all-games.Home',
        'team1_stats.win-pct-all-games.Away',
        'team2_stats.win-pct-all-games.Home',      # added
        'team2_stats.win-pct-all-games.Away',

        # On-base + slugging (OPS)
        'team1_stats.on-base-plus-slugging-pct.Home',
        'team1_stats.on-base-plus-slugging-pct.Away',  # added
        'team2_stats.on-base-plus-slugging-pct.Home',  # added
        'team2_stats.on-base-plus-slugging-pct.Away',

        # Slugging %
        'team1_stats.slugging-pct.Home',
        'team1_stats.slugging-pct.Away',            # added
        'team2_stats.slugging-pct.Home',            # added
        'team2_stats.slugging-pct.Away',

        # Batting average
        'team1_stats.batting-average.2025',
        'team1_stats.batting-average.Home',         # season & home
        'team1_stats.batting-average.Away',         # added
        'team2_stats.batting-average.2025',         # added
        'team2_stats.batting-average.Home',         # added
        'team2_stats.batting-average.Away',         # added

        # Total bases per game
        'team1_stats.total-bases-per-game.Home',
        'team1_stats.total-bases-per-game.Away',    # added
        'team2_stats.total-bases-per-game.Home',    # added
        'team2_stats.total-bases-per-game.Away',    # added

        # At-bats per game
        'team1_stats.at-bats-per-game.2025',
        'team1_stats.at-bats-per-game.Home',        # added
        'team1_stats.at-bats-per-game.Away',        # added
        'team2_stats.at-bats-per-game.2025',
        'team2_stats.at-bats-per-game.Home',        # added
        'team2_stats.at-bats-per-game.Away',        # added

        # Runs per game
        'team1_stats.runs-per-game.2025',           # added
        'team1_stats.runs-per-game.Home',
        'team1_stats.runs-per-game.Away',           # added
        'team2_stats.runs-per-game.2025',
        'team2_stats.runs-per-game.Home',           # added
        'team2_stats.runs-per-game.Away',

        # Hits per game
        'team1_stats.hits-per-game.Home',
        'team1_stats.hits-per-game.Away',           # added
        'team2_stats.hits-per-game.Home',           # added
        'team2_stats.hits-per-game.Away',

        # Opponent runs (pitching side)
        'team1_stats.opponent-runs-per-game.Home',
        'team1_stats.opponent-runs-per-game.Away',  # added
        'team2_stats.opponent-runs-per-game.Home',  # added
        'team2_stats.opponent-runs-per-game.Away',

        # Pitching quality: ERA & WHIP
        'team1_stats.earned-run-average.Home',
        'team1_stats.earned-run-average.Away',      # added
        'team2_stats.earned-run-average.Home',      # added
        'team2_stats.earned-run-average.Away',      # added
        'team1_stats.walks-plus-hits-per-inning-pitched.2025',
        'team1_stats.walks-plus-hits-per-inning-pitched.Home',
        'team1_stats.walks-plus-hits-per-inning-pitched.Away',  # added
        'team2_stats.walks-plus-hits-per-inning-pitched.2025',
        'team2_stats.walks-plus-hits-per-inning-pitched.Home',  # added
        'team2_stats.walks-plus-hits-per-inning-pitched.Away',  # added

        # K’s per 9 innings
        'team1_stats.strikeouts-per-9.Home',
        'team1_stats.strikeouts-per-9.Away',        # added
        'team2_stats.strikeouts-per-9.Home',        # added
        'team2_stats.strikeouts-per-9.Away',        # added

        # Recent form (last 3 / last 1)
        'team1_stats.on-base-plus-slugging-pct.Last 3',
        'team2_stats.on-base-plus-slugging-pct.Last 3',
        'team1_stats.slugging-pct.Last 1',
        'team2_stats.slugging-pct.Last 1',
        'team1_stats.runs-per-game.Last 3',
        'team2_stats.runs-per-game.Last 3',
    ]
    # ── CONFIG ─────────────────────────────────────
    MODEL_TYPE   = "logistic_regression"   # linear_regression, random_forest, xgboost, neural_network
    COLUMN       = "stats"    # stats or normalized_stats
    TRAIN_QUERY  = "SELECT * FROM games WHERE sport = 'MLB';"

    # build a name like "lr_norm_NBA_all" or e.g. "rf_nonorm_MLB_date_<_2025_05_26"
    MODEL_NAME = build_model_name(MODEL_TYPE, COLUMN, TRAIN_QUERY)

    # ── TRAIN ─────────────────────────────────────
    model = MLModel(MODEL_NAME, MODEL_TYPE, column=COLUMN, use_random_subset_of_features=False, subset_fraction=0.025, feature_allowlist=features)
    model.train(TRAIN_QUERY)

if __name__ == "__main__":
    main()
