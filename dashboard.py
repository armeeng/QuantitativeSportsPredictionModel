# Save this file as dashboard.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import date
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
    # NEW: Import the Pregame class to access its update methods
    from Pregame import Pregame
    # --- NEW: Import the TestModel class ---
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

# --- Helper Functions (Unchanged) ----------------------------------------------------
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
    """Queries the database to get all games and their latest predictions for a given day."""
    # (This function remains unchanged)
    query = """
    SELECT
        g.game_id, g.date, g.sport, g.team1_id, g.team1_name, g.team1_logo,
        g.team2_id, g.team2_name, g.team2_logo, g.team1_moneyline, g.team2_moneyline,
        g.team1_spread, g.team1_spread_odds, g.team2_spread_odds, g.total_score,
        g.over_odds, g.under_odds, g.team1_score, g.team2_score, -- <-- NEWLY ADDED
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
        
# --- NEW: Function to load all completed games for historical analysis ---
# --- NEW: Function to load all completed games for historical analysis ---
@st.cache_data(ttl=3600)
def load_historical_data_for_testmodel(selected_sport: str, selected_model: str) -> pd.DataFrame:
    """
    Queries the database for all COMPLETED games with final scores and predictions
    for a given sport and model to be used as a test set.
    """
    # CORRECTED SQL QUERY: Uses 'team1_score' and 'team2_score' and cleans up the problematic comment.
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
        AND p.created_at = (
            SELECT MAX(created_at) FROM predictions p2
            WHERE p2.game_id = p.game_id AND p2.model_name = p.model_name
        )
    """
    params = {"selected_sport": selected_sport, "selected_model": selected_model}
    try:
        df = pd.read_sql_query(query, engine, params=params)
        if not df.empty and 'prediction_data' in df.columns:
            def safe_json_load(j):
                try: return json.loads(j) if isinstance(j, str) else j
                except (json.JSONDecodeError, TypeError): return {}
            # Apply the safe loader and then normalize
            parsed_data = df['prediction_data'].apply(safe_json_load)
            pred_df = pd.json_normalize(parsed_data.tolist()).reindex(df.index)
            df = df.drop(columns=['prediction_data']).join(pred_df)
        return df
    except Exception as e:
        st.error(f"Failed to load historical data from the database: {e}")
        return pd.DataFrame()


def fetch_live_scoreboard_data(selected_date: date, selected_sport: str) -> dict:
    """Hits the ESPN scoreboard API to get live status and scores for all games on a given day."""
    # (This function remains unchanged)
    live_data_map = {}
    ESPN_MAP = {'NBA': ('basketball', 'nba'), 'NFL': ('football', 'nfl'), 'CFB': ('football', 'college-football'), 'CBB': ('basketball', 'mens-college-basketball'), 'MLB': ('baseball', 'mlb')}
    
    if selected_sport not in ESPN_MAP:
        return {}

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
                live_data_map[game_id] = {
                    'away_score': int(away.get("score", 0)),
                    'home_score': int(home.get("score", 0)),
                    'status_detail': status.get("detail", "Scheduled"),
                }
    except requests.exceptions.RequestException as e:
        st.toast(f"Couldn't fetch live scores: {e}", icon="📡")
    
    return live_data_map

@st.cache_data
def get_filter_options():
    sports = pd.read_sql_query("SELECT DISTINCT sport FROM games ORDER BY sport", engine)
    models = pd.read_sql_query("SELECT DISTINCT model_name FROM predictions ORDER BY model_name", engine)
    return sports['sport'].tolist(), models['model_name'].tolist()

# --- UI Layout -----------------------------------------------------------------------
st.title(f"🏈 ⚾️ {APP_TITLE} 🏀 🏒")

with st.sidebar:
    st.header("Filters")
    available_sports, available_models = get_filter_options()
    if not available_sports or not available_models:
        st.warning("No data in the database.")
    else:
        selected_sport = st.selectbox("Select Sport", available_sports)
        selected_date = st.date_input("Select Date", date.today())
        selected_model = st.selectbox("Select Model", available_models)

        # --- NEW: Refresh Button ---
        st.divider()
        if st.button("✍️ Update Final Scores & Odds", use_container_width=True, help="Run this after games are final to save scores and closing odds to the database."):
            with st.spinner(f"Updating {selected_sport} on {selected_date}..."):
                try:
                    # Instantiate Pregame and run the update
                    pg = Pregame(date=selected_date, sport=selected_sport)
                    pg.update_final_scores_and_closing_odds()
                    st.cache_data.clear() # Clear cache to get fresh data
                    st.success("Database updated successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"An error occurred during refresh: {e}")


# --- NEW: Create tabs for daily bets vs historical analysis ---
tab1, tab2 = st.tabs(["Today's Betting Card", "Historical Performance Analysis"])


with tab1:
    st.markdown("### Betting Strategy Configuration")
    c1, c2 = st.columns(2)
    bankroll = c1.number_input("Enter your bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
    max_bet_percent = c2.number_input("Max Bet as % of Bankroll (The 'Safety Net')", min_value=0.1, max_value=100.0, value=1.0, step=0.1,
                                    help="Sets the MAXIMUM percentage of your bankroll to risk on any single bet, mirroring the 'max_fraction' in the backtest.")
    max_bet_fraction = max_bet_percent / 100.0

    # --- Main Content Area (for daily games) ---
    if 'selected_sport' in locals() and 'selected_model' in locals():
        df = load_data(selected_date, selected_sport, selected_model)
        live_scoreboard = fetch_live_scoreboard_data(selected_date, selected_sport)

        if df.empty:
            st.warning(f"No games or predictions found for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}** with model **{selected_model}**.")
        else:
            st.success(f"Found {len(df)} games for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}**")
            for index, game in df.iterrows():
                # This entire loop is the original dashboard content, now nested in the tab
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
                    st.markdown("##### Point Spread")
                    # (Spread and Total logic is identical, just nested)
                    # ... [omitted for brevity, it's the same as your original file] ...

            with st.expander("Show Raw Data Table"):
                st.dataframe(df)
    else:
        st.info("Please select filters from the sidebar to view games.")

# --- NEW: Code for the Historical Performance Analysis Tab ---
with tab2:
    st.header(f"Historical Performance Review")
    st.markdown("This section analyzes the performance of the selected model across all **completed games** found in the database.")

    if 'selected_sport' in locals() and 'selected_model' in locals():
        with st.spinner(f"Loading and analyzing historical data for model '{selected_model}' in '{selected_sport}'..."):
            hist_df = load_historical_data_for_testmodel(selected_sport, selected_model)

            # CORRECTED: Check for 'team1_score' instead of 'team1_final_score'
            if hist_df.empty or 'team1_score' not in hist_df.columns:
                st.warning(f"No completed games with final scores found for model **'{selected_model}'** in **{selected_sport}**. Run the 'Update Final Scores' process on the sidebar for past dates to enable this analysis.")
            elif not all(k in hist_df.columns for k in ['team1_win_prob', 'team1_cover_prob', 'over_prob']):
                 st.error("Historical data is missing required prediction probabilities. The selected model may not be a classifier, which is required for this analysis.")
            else:
                st.success(f"Found and analyzed **{len(hist_df)}** completed games.")

                # 1. Prepare data for the TestModel class
                # CORRECTED: Use 'team1_score' and 'team2_score' to create the y_test numpy array.
                y_test = hist_df[['team1_score', 'team2_score']].to_numpy()

                # test_odds: The dataframe itself contains all the odds columns
                # Ensure correct column names as expected by TestModel
                hist_df.rename(columns={
                    'team1_moneyline': 'team1_ml',
                    'team2_moneyline': 'team2_ml',
                }, inplace=True, errors='ignore')

                # predictions: A dictionary of probabilities
                predictions = {
                    'win': hist_df[['team1_win_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(),
                    'spread': hist_df[['team1_cover_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(),
                    'over': hist_df[['over_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list()
                }
                # Convert lists of lists to numpy arrays
                for key in predictions:
                    predictions[key] = pd.DataFrame(predictions[key]).to_numpy()
                
                # 2. Instantiate the TestModel class
                try:
                    analyzer = TestModel(predictions=predictions, y_test=y_test, test_odds=hist_df)

                    # 3. Call methods and display results in Streamlit
                    st.subheader("Prediction Accuracy")
                    acc = analyzer.calculate_accuracies()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Winner Accuracy", f"{acc['win_accuracy']:.2%}", f"{acc['correct_winner_preds']}/{acc['total_games']}")
                    c2.metric("Spread Accuracy", f"{acc['spread_accuracy']:.2%}", f"{acc['correct_spread_preds']}/{acc['num_spread_outcomes']}")
                    c3.metric("Total Accuracy", f"{acc['total_accuracy']:.2%}", f"{acc['correct_ou_preds']}/{acc['num_ou_outcomes']}")

                    st.subheader("Profitability Analysis")
                    pnl_flat = analyzer.calculate_pnl_of_all_games()
                    pnl_ev = analyzer.calculate_pnl_of_game_above_ev_threshold()
                    
                    df_pnl = pd.DataFrame({
                        'Bet Type': ['Moneyline', 'Spread', 'Over/Under'],
                        'Flat Bet PnL': [f"${pnl_flat['moneyline_pnl']:.2f}", f"${pnl_flat['spread_pnl']:.2f}", f"${pnl_flat['ou_pnl']:.2f}"],
                        'Flat Bets Placed': [pnl_flat['moneyline_bets_placed'], pnl_flat['spread_bets_placed'], pnl_flat['ou_bets_placed']],
                        '+EV Bets PnL': [f"${pnl_ev['moneyline']['pnl']:.2f}", f"${pnl_ev['spread']['pnl']:.2f}", f"${pnl_ev['ou']['pnl']:.2f}"],
                        '+EV Bets Placed': [pnl_ev['moneyline']['count'], pnl_ev['spread']['count'], pnl_ev['ou']['count']]
                    }).set_index('Bet Type')
                    st.dataframe(df_pnl, use_container_width=True)

                    # Capture printed output for p-values
                    st.subheader("Statistical Significance (P-Values)")
                    f = io.StringIO()
                    with redirect_stdout(f):
                        analyzer.calculate_p_values()
                    p_value_output = f.getvalue()
                    st.code(p_value_output)

                    st.subheader("Kelly Criterion Historical Backtest")
                    kelly_results = analyzer.simulate_kelly_betting(initial_bankroll=1000)
                    st.json(kelly_results)

                    # Capture plots and printed text for the remaining functions
                    analysis_functions = {
                        "Model vs. Market Calibration": analyzer.check_calibration,
                        "Probabilistic Monte Carlo (Model Probs)": analyzer.run_probabilistic_monte_carlo,
                        "Probabilistic Monte Carlo (Market Probs)": analyzer.run_market_monte_carlo,
                        "Bootstrap Simulation (from historical bets)": analyzer.run_bootstrap_simulation
                    }

                    for title, func in analysis_functions.items():
                        st.subheader(title)
                        f = io.StringIO()
                        with redirect_stdout(f):
                            # The plotting functions in TestModel also print results, we capture them
                            plt.close('all') # Close previous figures
                            func()
                            fig = plt.gcf() # Get the current figure generated by the function
                        
                        # Display the captured text and the plot
                        st.code(f.getvalue())
                        if fig.get_axes(): # Check if the figure has anything drawn on it
                            st.pyplot(fig)
                        plt.close(fig) # Close the figure to free memory

                except Exception as e:
                    st.error(f"An error occurred during historical analysis with the TestModel class.")
                    st.exception(e)
    else:
        st.info("Please select filters from the sidebar to run the analysis.")