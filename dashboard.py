# Save this file as dashboard.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import date, timedelta
import os
import json
import time
import sys
import requests
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import io

# --- Path Correction & Model Imports --------------------------------------------------
# This allows the script to find your other Python files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from Pregame import Pregame
    from TestModel import TestModel
except ImportError as e:
    st.error(f"FATAL: Could not import a required class. Ensure Pregame.py and TestModel.py are accessible.")
    st.exception(e) # Show the full traceback for debugging
    st.stop()


# --- Configuration -------------------------------------------------------------------
APP_TITLE = "Sports Predictions Dashboard"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Database Connection Setup -------------------------------------------------------
DATABASE_URL = f"sqlite:///sports.db"
engine = create_engine(DATABASE_URL)

# --- Helper Functions ----------------------------------------------------
def american_to_decimal(odds: float) -> float:
    if pd.isna(odds) or odds == 0: return None
    if odds > 0: return (odds / 100) + 1
    else: return (100 / abs(odds)) + 1

def american_to_prob(odds: float) -> float:
    if pd.isna(odds): return None
    if odds > 0: return 100 / (odds + 100)
    else: return abs(odds) / (abs(odds) + 100)

def calculate_ev(prob_win: float, odds: float) -> float:
    if pd.isna(prob_win) or pd.isna(odds): return 0
    decimal_odds = american_to_decimal(odds)
    if decimal_odds is None: return 0
    return (prob_win * (decimal_odds - 1)) - ((1 - prob_win) * 1)

def calculate_kelly_fraction(prob_win: float, decimal_odds: float) -> float:
    if pd.isna(prob_win) or pd.isna(decimal_odds) or decimal_odds <= 1: return 0
    p, q, b = prob_win, 1 - prob_win, decimal_odds - 1
    if (p * b - q) <= 0: return 0
    return (p * b - q) / b

# --- Data Loading -------------------------------------------------------------------
@st.cache_data(ttl=600)
def load_data(selected_date: date, selected_sport: str, selected_model: str) -> pd.DataFrame:
    query = """
    SELECT
        g.game_id, g.date, g.sport, g.team1_id, g.team1_name, g.team1_logo,
        g.team2_id, g.team2_name, g.team2_logo, g.team1_moneyline, g.team2_moneyline,
        g.team1_spread, g.team1_spread_odds, g.team2_spread_odds, g.total_score,
        g.over_odds, g.under_odds, g.team1_score, g.team2_score,
        p.model_name, p.prediction_data
    FROM games g
    JOIN predictions p ON g.game_id = p.game_id
    WHERE
        g.date = :selected_date
        AND g.sport = :selected_sport
        AND p.model_name = :selected_model
        AND p.created_at = (
            SELECT MAX(created_at) FROM predictions p2
            WHERE p2.game_id = p.game_id AND p2.model_name = p.model_name
        )
    """
    params = {"selected_date": selected_date, "selected_sport": selected_sport, "selected_model": selected_model}
    try:
        df = pd.read_sql_query(query, engine, params=params)
        if not df.empty and 'prediction_data' in df.columns:
            def safe_json_load(j):
                if isinstance(j, str):
                    try: return json.loads(j)
                    except json.JSONDecodeError: return None
                return j
            parsed_data = df['prediction_data'].apply(safe_json_load)
            pred_df = pd.json_normalize(parsed_data)
            df = df.drop(columns=['prediction_data']).join(pred_df)
        return df
    except Exception as e:
        st.error(f"Failed to load data from the database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_historical_data_for_testmodel(selected_sport: str, selected_model: str, start_date: date, end_date: date) -> pd.DataFrame:
    query = """
    SELECT
        g.*,
        p.prediction_data
    FROM games g
    JOIN predictions p ON g.game_id = p.game_id
    WHERE
        g.team1_score IS NOT NULL
        AND g.team2_score IS NOT NULL
        AND g.sport = :selected_sport
        AND p.model_name = :selected_model
        AND g.date BETWEEN :start_date AND :end_date
        AND p.created_at = (
            SELECT MAX(created_at) FROM predictions p2
            WHERE p2.game_id = p.game_id AND p2.model_name = p.model_name
        )
    """
    params = {
        "selected_sport": selected_sport,
        "selected_model": selected_model,
        "start_date": start_date,
        "end_date": end_date
    }
    try:
        df = pd.read_sql_query(query, engine, params=params)
        if not df.empty and 'prediction_data' in df.columns:
            def safe_json_load(j):
                try: return json.loads(j) if isinstance(j, str) else j
                except (json.JSONDecodeError, TypeError): return {}
            parsed_data = df['prediction_data'].apply(safe_json_load)
            pred_df = pd.json_normalize(parsed_data.tolist()).reindex(df.index)
            df = df.drop(columns=['prediction_data']).join(pred_df)
        return df
    except Exception as e:
        st.error(f"Failed to load historical data from the database: {e}")
        return pd.DataFrame()

def fetch_live_scoreboard_data(selected_date: date, selected_sport: str) -> dict:
    live_data_map = {}
    ESPN_MAP = {'NBA': ('basketball', 'nba'), 'NFL': ('football', 'nfl'), 'CFB': ('football', 'college-football'), 'CBB': ('basketball', 'mens-college-basketball'), 'MLB': ('baseball', 'mlb')}
    if selected_sport not in ESPN_MAP: return {}
    category, league = ESPN_MAP[selected_sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{category}/{league}/scoreboard"
    date_str = selected_date.strftime("%Y%m%d")
    try:
        resp = requests.get(url, params={"dates": date_str}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        for event in data.get("events", []):
            game_id = event.get("id")
            if not game_id: continue
            comp = event["competitions"][0]
            status = comp.get("status", {}).get("type", {})
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away, home = teams.get("away"), teams.get("home")
            if away and home:
                live_data_map[game_id] = {'away_score': int(away.get("score", 0)), 'home_score': int(home.get("score", 0)), 'status_detail': status.get("detail", "Scheduled")}
    except requests.exceptions.RequestException as e:
        st.toast(f"Couldn't fetch live scores: {e}", icon="📡")
    return live_data_map

@st.cache_data
def get_filter_options():
    sports = pd.read_sql_query("SELECT DISTINCT sport FROM games ORDER BY sport", engine)
    models = pd.read_sql_query("SELECT DISTINCT model_name FROM predictions ORDER BY model_name", engine)
    return sports['sport'].tolist(), models['model_name'].tolist()

@st.cache_data(ttl=3600)
def get_historical_date_range(selected_sport: str) -> tuple[date, date]:
    query = "SELECT MIN(date), MAX(date) FROM games WHERE sport = :selected_sport AND team1_score IS NOT NULL AND team2_score IS NOT NULL"
    params = {"selected_sport": selected_sport}
    try:
        result_df = pd.read_sql_query(query, engine, params=params)
        if not result_df.empty:
            min_date_str, max_date_str = result_df.iloc[0]
            min_date = pd.to_datetime(min_date_str).date() if min_date_str else date.today()
            max_date = pd.to_datetime(max_date_str).date() if max_date_str else date.today()
            return min_date, max_date
    except Exception:
        pass
    return date.today() - timedelta(days=30), date.today()

# --- UI Layout -----------------------------------------------------------------------
st.title(f"🏈 ⚾️ {APP_TITLE} 🏀 🏒")

with st.sidebar:
    st.header("Filters")
    available_sports, available_models = get_filter_options()
    if not available_sports or not available_models:
        st.warning("No data in the database.")
    else:
        selected_sport = st.selectbox("Select Sport", available_sports)
        selected_date = st.date_input("Select Date (for Betting Card)", date.today())
        selected_model = st.selectbox("Select Model", available_models)
        st.divider()
        if st.button("✍️ Update Final Scores & Odds", use_container_width=True, help="Run this after games are final to save scores and closing odds to the database."):
            with st.spinner(f"Updating {selected_sport} on {selected_date}..."):
                try:
                    pg = Pregame(date=selected_date, sport=selected_sport)
                    pg.update_final_scores_and_closing_odds()
                    st.cache_data.clear()
                    st.success("Database updated successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during refresh: {e}")

st.markdown("### Betting Strategy Configuration")
st.markdown("These settings apply to both the **Betting Card** and the **Historical Performance** backtest.")
c1, c2 = st.columns(2)
bankroll = c1.number_input("Enter your bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
max_bet_percent = c2.number_input("Max Bet as % of Bankroll (The 'Safety Net')", min_value=0.1, max_value=100.0, value=1.0, step=0.1, help="Sets the MAXIMUM percentage of your bankroll to risk on any single bet, mirroring the 'max_fraction' in the backtest.")
max_bet_fraction = max_bet_percent / 100.0

tab1, tab2 = st.tabs(["Today's Betting Card", "Historical Performance Analysis"])

with tab1:
    if 'selected_sport' in locals() and 'selected_model' in locals():
        df = load_data(selected_date, selected_sport, selected_model)
        live_scoreboard = fetch_live_scoreboard_data(selected_date, selected_sport)
        if df.empty:
            st.warning(f"No games or predictions found for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}** with model **{selected_model}**.")
        else:
            st.success(f"Found {len(df)} games for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}**")
            for index, game in df.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2.5, 1.5, 2.5])
                    with col1:
                        st.image(game['team1_logo'], width=60)
                        st.subheader(f"{game['team1_name']} (Away)")
                    with col2:
                        live_data = live_scoreboard.get(str(game['game_id']), {})
                        live_t1_score = live_data.get('away_score')
                        live_t2_score = live_data.get('home_score')
                        status_text = live_data.get('status_detail', 'Scheduled')
                        if live_t1_score is not None and live_t2_score is not None:
                            st.metric(label=status_text, value=f"{live_t1_score} – {live_t2_score}", label_visibility="visible")
                        else:
                            st.markdown("<h3 style='text-align: center; color: grey;'>VS</h3>", unsafe_allow_html=True)
                    with col3:
                        st.image(game['team2_logo'], width=60)
                        st.subheader(f"{game['team2_name']} (Home)")

                    st.divider()

                    # --- Moneyline Section ---
                    st.markdown("##### Moneyline")
                    b1, b2 = st.columns([1.5, 2.5])
                    with b1:
                        prob_t1, odds_t1 = game.get('team1_win_prob'), game.get('team1_moneyline')
                        dec_odds_t1, ev_t1 = american_to_decimal(odds_t1), calculate_ev(prob_t1, odds_t1)
                        kelly_t1 = calculate_kelly_fraction(prob_t1, dec_odds_t1)
                        prob_t2 = 1 - prob_t1 if prob_t1 is not None else None
                        odds_t2, dec_odds_t2 = game.get('team2_moneyline'), american_to_decimal(game.get('team2_moneyline'))
                        ev_t2, kelly_t2 = calculate_ev(prob_t2, odds_t2), calculate_kelly_fraction(prob_t2, dec_odds_t2)
                        if ev_t1 > 0 and ev_t1 > ev_t2:
                            bet_fraction, bet_size = min(kelly_t1, max_bet_fraction), bankroll * min(kelly_t1, max_bet_fraction)
                            st.success(f"✅ Bet on {game['team1_name']} ({odds_t1:+.0f})")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t1*100:.2f}%")
                        elif ev_t2 > 0 and ev_t2 > ev_t1:
                            bet_fraction, bet_size = min(kelly_t2, max_bet_fraction), bankroll * min(kelly_t2, max_bet_fraction)
                            st.success(f"✅ Bet on {game['team2_name']} ({odds_t2:+.0f})")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t2*100:.2f}%")
                        else: st.info("No value found. Do not bet.")
                    with b2:
                        imp_prob_t1, imp_prob_t2 = american_to_prob(odds_t1), american_to_prob(odds_t2)
                        st.dataframe({'Team': [game['team1_name'], game['team2_name']], 'Model Prob': [f"{prob_t1*100:.1f}%" if prob_t1 else 'N/A', f"{prob_t2*100:.1f}%" if prob_t2 else 'N/A'], 'Odds': [odds_t1, odds_t2], 'Implied Prob': [f"{imp_prob_t1*100:.1f}%" if imp_prob_t1 else 'N/A', f"{imp_prob_t2*100:.1f}%" if imp_prob_t2 else 'N/A'], 'EV': [f"{ev_t1*100:.2f}%", f"{ev_t2*100:.2f}%"]}, use_container_width=True)

                    st.divider()

                    # --- Point Spread Section ---
                    st.markdown("##### Point Spread")
                    b1, b2 = st.columns([1.5, 2.5])
                    with b1:
                        prob_t1_cover, spread_t1 = game.get('team1_cover_prob'), game.get('team1_spread')
                        odds_t1_spread, odds_t2_spread = game.get('team1_spread_odds'), game.get('team2_spread_odds')
                        dec_odds_t1_spread, ev_t1_spread = american_to_decimal(odds_t1_spread), calculate_ev(prob_t1_cover, odds_t1_spread)
                        kelly_t1_spread = calculate_kelly_fraction(prob_t1_cover, dec_odds_t1_spread)
                        
                        prob_t2_cover = 1 - prob_t1_cover if prob_t1_cover is not None else None
                        dec_odds_t2_spread, ev_t2_spread = american_to_decimal(odds_t2_spread), calculate_ev(prob_t2_cover, odds_t2_spread)
                        kelly_t2_spread = calculate_kelly_fraction(prob_t2_cover, dec_odds_t2_spread)

                        if ev_t1_spread > 0 and ev_t1_spread > ev_t2_spread:
                            bet_fraction, bet_size = min(kelly_t1_spread, max_bet_fraction), bankroll * min(kelly_t1_spread, max_bet_fraction)
                            st.success(f"✅ Bet on {game['team1_name']} ({spread_t1:+.1f})")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t1_spread*100:.2f}%")
                        elif ev_t2_spread > 0 and ev_t2_spread > ev_t1_spread:
                            bet_fraction, bet_size = min(kelly_t2_spread, max_bet_fraction), bankroll * min(kelly_t2_spread, max_bet_fraction)
                            st.success(f"✅ Bet on {game['team2_name']} ({-spread_t1:+.1f})")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t2_spread*100:.2f}%")
                        else: st.info("No value found. Do not bet.")
                    with b2:
                        imp_prob_t1_spread, imp_prob_t2_spread = american_to_prob(odds_t1_spread), american_to_prob(odds_t2_spread)
                        st.dataframe({
                            'Bet': [f"{game['team1_name']} ({spread_t1:+.1f})", f"{game['team2_name']} ({-spread_t1:+.1f})"],
                            'Model Prob': [f"{prob_t1_cover*100:.1f}%" if prob_t1_cover else 'N/A', f"{prob_t2_cover*100:.1f}%" if prob_t2_cover else 'N/A'],
                            'Odds': [odds_t1_spread, odds_t2_spread],
                            'Implied Prob': [f"{imp_prob_t1_spread*100:.1f}%" if imp_prob_t1_spread else 'N/A', f"{imp_prob_t2_spread*100:.1f}%" if imp_prob_t2_spread else 'N/A'],
                            'EV': [f"{ev_t1_spread*100:.2f}%", f"{ev_t2_spread*100:.2f}%"]
                        }, use_container_width=True)

                    st.divider()
                    
                    # --- Totals (Over/Under) Section ---
                    st.markdown("##### Totals (Over/Under)")
                    b1, b2 = st.columns([1.5, 2.5])
                    with b1:
                        prob_over, total_line = game.get('over_prob'), game.get('total_score')
                        over_odds, under_odds = game.get('over_odds'), game.get('under_odds')
                        dec_odds_over, ev_over = american_to_decimal(over_odds), calculate_ev(prob_over, over_odds)
                        kelly_over = calculate_kelly_fraction(prob_over, dec_odds_over)
                        
                        prob_under = 1 - prob_over if prob_over is not None else None
                        dec_odds_under, ev_under = american_to_decimal(under_odds), calculate_ev(prob_under, under_odds)
                        kelly_under = calculate_kelly_fraction(prob_under, dec_odds_under)

                        if ev_over > 0 and ev_over > ev_under:
                            bet_fraction, bet_size = min(kelly_over, max_bet_fraction), bankroll * min(kelly_over, max_bet_fraction)
                            st.success(f"✅ Bet on Over {total_line}")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_over*100:.2f}%")
                        elif ev_under > 0 and ev_under > ev_over:
                            bet_fraction, bet_size = min(kelly_under, max_bet_fraction), bankroll * min(kelly_under, max_bet_fraction)
                            st.success(f"✅ Bet on Under {total_line}")
                            st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_under*100:.2f}%")
                        else: st.info("No value found. Do not bet.")
                    with b2:
                        imp_prob_over, imp_prob_under = american_to_prob(over_odds), american_to_prob(under_odds)
                        st.dataframe({
                            'Bet': [f"Over {total_line}", f"Under {total_line}"],
                            'Model Prob': [f"{prob_over*100:.1f}%" if prob_over else 'N/A', f"{prob_under*100:.1f}%" if prob_under else 'N/A'],
                            'Odds': [over_odds, under_odds],
                            'Implied Prob': [f"{imp_prob_over*100:.1f}%" if imp_prob_over else 'N/A', f"{imp_prob_under*100:.1f}%" if imp_prob_under else 'N/A'],
                            'EV': [f"{ev_over*100:.2f}%", f"{ev_under*100:.2f}%"]
                        }, use_container_width=True)

            with st.expander("Show Raw Data Table"):
                st.dataframe(df)
    else:
        st.info("Please select filters from the sidebar to view games.")

with tab2:
    st.header(f"Historical Performance Review")
    st.markdown("This section runs all tests from the `TestModel` class and displays the raw output.")

    if 'selected_sport' in locals() and 'selected_model' in locals():
        min_hist_date, max_hist_date = get_historical_date_range(selected_sport)
        st.markdown("#### Select Date Range for Analysis")
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date", min_hist_date, min_value=min_hist_date, max_value=max_hist_date)
        end_date = c2.date_input("End Date", max_hist_date, min_value=min_hist_date, max_value=max_hist_date)

        if start_date > end_date:
            st.error("Error: Start date cannot be after end date.")
        else:
            st.markdown("---")
            with st.spinner(f"Loading and analyzing historical data for model '{selected_model}' in '{selected_sport}'..."):
                hist_df = load_historical_data_for_testmodel(selected_sport, selected_model, start_date, end_date)

                if hist_df.empty or 'team1_score' not in hist_df.columns:
                    st.warning(f"No completed games with final scores found for model **'{selected_model}'** in **{selected_sport}** within the selected date range. Try expanding the date range or run the 'Update Final Scores' process for past dates.")
                elif not all(k in hist_df.columns for k in ['team1_win_prob', 'team1_cover_prob', 'over_prob']):
                     st.error("Historical data is missing required prediction probabilities. The selected model may not be a classifier, which is required for this analysis.")
                else:
                    st.success(f"Found **{len(hist_df)}** completed games to analyze from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}**.")

                    y_test = hist_df[['team1_score', 'team2_score']].to_numpy()
                    hist_df.rename(columns={'team1_moneyline': 'team1_ml', 'team2_moneyline': 'team2_ml'}, inplace=True, errors='ignore')
                    predictions = {'win': hist_df[['team1_win_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(), 'spread': hist_df[['team1_cover_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(), 'over': hist_df[['over_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list()}
                    for key in predictions: predictions[key] = pd.DataFrame(predictions[key]).to_numpy()

                    try:
                        analyzer = TestModel(predictions=predictions, y_test=y_test, test_odds=hist_df)
                        
                        st.subheader("Full Performance Analysis Report")
                        
                        # --- Manually call text-based analysis functions ---
                        text_output = io.StringIO()
                        with redirect_stdout(text_output):
                            acc = analyzer.calculate_accuracies()
                            print("\nModel Prediction Accuracy:")
                            print(f"  - Winner Accuracy:     {acc['win_accuracy']:.2%} ({acc['correct_winner_preds']}/{acc['total_games']})")
                            print(f"  - Spread Accuracy:     {acc['spread_accuracy']:.2%} ({acc['correct_spread_preds']}/{acc['num_spread_outcomes']})")
                            print(f"  - Over/Under Accuracy: {acc['total_accuracy']:.2%} ({acc['correct_ou_preds']}/{acc['num_ou_outcomes']})")

                            pnl = analyzer.calculate_pnl_of_all_games()
                            print(f"\nProfit & Loss (flat $1 bets on all available odds):")
                            print(f"  - Moneyline PnL:      ${pnl['moneyline_pnl']:.2f} from {pnl['moneyline_bets_placed']} bets")
                            print(f"  - Spread PnL:         ${pnl['spread_pnl']:.2f} from {pnl['spread_bets_placed']} bets")
                            print(f"  - Over/Under PnL:     ${pnl['ou_pnl']:.2f} from {pnl['ou_bets_placed']} bets")

                            if isinstance(analyzer.predictions, dict):
                                ev_pnl = analyzer.calculate_pnl_of_game_above_ev_threshold()
                                ml_info, spread_info, ou_info = ev_pnl['moneyline'], ev_pnl['spread'], ev_pnl['ou']
                                print("\nPnL on +EV Bets (Classifier Only):")
                                print(f"  - Moneyline:  ${ml_info['pnl']:.2f} from {ml_info['count']} bets ({ml_info['count']/acc['total_games']:.1%})")
                                print(f"  - Spread:     ${spread_info['pnl']:.2f} from {spread_info['count']} bets ({spread_info['count']/acc['total_games']:.1%})")
                                print(f"  - Over/Under: ${ou_info['pnl']:.2f} from {ou_info['count']} bets ({ou_info['count']/acc['total_games']:.1%})")

                                analyzer.calculate_p_values()

                                kelly_results = analyzer.simulate_kelly_betting(initial_bankroll=bankroll, max_fraction=max_bet_fraction)
                                kelly_ml, kelly_spread, kelly_ou = kelly_results['moneyline'], kelly_results['spread'], kelly_results['ou']
                                print(f"\nKelly Criterion Simulation (Historical Backtest, max_frac={max_bet_fraction:.2%}):")
                                print(f"  - Moneyline:  Final Bankroll: ${kelly_ml['final_bankroll']:.2f} (Profit: ${kelly_ml['final_bankroll'] - bankroll:.2f})")
                                print(f"  - Spread:     Final Bankroll: ${kelly_spread['final_bankroll']:.2f} (Profit: ${kelly_spread['final_bankroll'] - bankroll:.2f})")
                                print(f"  - Over/Under: Final Bankroll: ${kelly_ou['final_bankroll']:.2f} (Profit: ${kelly_ou['final_bankroll'] - bankroll:.2f})")
                        
                        st.code(text_output.getvalue(), language='text')

                        # --- Manually call functions that generate plots ---
                        if isinstance(analyzer.predictions, dict):
                            # Use lambdas to pass arguments to functions that need them
                            plotting_functions = {
                                "Model vs. Market Calibration": analyzer.check_calibration,
                                "Probabilistic Monte Carlo (Model Probs)": lambda: analyzer.run_probabilistic_monte_carlo(initial_bankroll=bankroll, max_fraction=max_bet_fraction),
                                "Probabilistic Monte Carlo (Market Probs)": lambda: analyzer.run_market_monte_carlo(initial_bankroll=bankroll, max_fraction=max_bet_fraction),
                                "Bootstrap Simulation": lambda: analyzer.run_bootstrap_simulation(initial_bankroll=bankroll, max_fraction=max_bet_fraction)
                            }

                            for title, func in plotting_functions.items():
                                st.subheader(title)
                                plot_output_capture = io.StringIO()
                                with redirect_stdout(plot_output_capture):
                                    plt.close('all')
                                    func()
                                    fig = plt.gcf()
                                st.code(plot_output_capture.getvalue(), language='text')
                                if fig.get_axes(): st.pyplot(fig)
                                plt.close(fig)
                    
                    except ValueError as e:
                        if 'y_true contains only one label' in str(e):
                            st.warning("⚠️ **Could not generate full report.**", icon="⚠️")
                            st.info("This is likely because all game outcomes in the selected date range were the same. The analysis requires at least one of each outcome (e.g., at least one win and one loss) to run.")
                        else: st.error("An unexpected value error occurred during the analysis."); st.exception(e)
                    except Exception as e: st.error("An error occurred during historical analysis."); st.exception(e)
    else:
        st.info("Please select filters from the sidebar to run the analysis.")