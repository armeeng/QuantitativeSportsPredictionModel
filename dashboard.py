# Save this file as dashboard.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from datetime import date
import os
import json

# --- Configuration -------------------------------------------------------------------
APP_TITLE = "Sports Predictions Dashboard"
st.set_page_config(page_title=APP_TITLE, layout="wide")

# --- Database Connection Setup -------------------------------------------------------
# Assumes the script is in a subdirectory and the db is in the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
DATABASE_URL = f"sqlite:///sports.db"
engine = create_engine(DATABASE_URL)

# --- Helper Functions (from your TestModel class) ------------------------------------

def american_to_decimal(odds: float) -> float:
    """Converts American odds to decimal odds."""
    if pd.isna(odds) or odds == 0: return None
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def american_to_prob(odds: float) -> float:
    """Converts American odds to an implied probability."""
    if pd.isna(odds):
        return None
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def calculate_ev(prob_win: float, odds: float) -> float:
    """Calculates the Expected Value of a bet."""
    if pd.isna(prob_win) or pd.isna(odds): return 0
    
    decimal_odds = american_to_decimal(odds)
    if decimal_odds is None: return 0

    profit_if_win = decimal_odds - 1
    loss_if_loss = 1
    prob_loss = 1 - prob_win

    return (prob_win * profit_if_win) - (prob_loss * loss_if_loss)

def calculate_kelly_fraction(prob_win: float, decimal_odds: float) -> float:
    """Calculates the Kelly Criterion fraction of bankroll to bet."""
    if pd.isna(prob_win) or pd.isna(decimal_odds) or decimal_odds <= 1: return 0
    
    p = prob_win
    q = 1 - p
    b = decimal_odds - 1 # Net odds
    
    # If edge is not positive, fraction is 0
    if (p * b - q) <= 0: return 0
    
    return (p * b - q) / b

# --- Data Loading -------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_data(selected_date: date, selected_sport: str, selected_model: str) -> pd.DataFrame:
    """Queries the database to get all games and their latest predictions for a given day."""
    query = """
    SELECT
        g.game_id, g.date, g.sport, g.team1_id, g.team1_name, g.team1_logo,
        g.team2_id, g.team2_name, g.team2_logo, g.team1_moneyline, g.team2_moneyline,
        g.team1_spread, g.team1_spread_odds, g.team2_spread_odds, g.total_score,
        g.over_odds, g.under_odds, p.model_name, p.prediction_data
    FROM games g
    JOIN predictions p ON g.game_id = p.game_id
    WHERE
        g.date = :selected_date
        AND g.sport = :selected_sport
        AND p.model_name = :selected_model
        AND p.created_at = (
            SELECT MAX(created_at)
            FROM predictions p2
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

@st.cache_data
def get_filter_options():
    sports = pd.read_sql_query("SELECT DISTINCT sport FROM games ORDER BY sport", engine)
    models = pd.read_sql_query("SELECT DISTINCT model_name FROM predictions ORDER BY model_name", engine)
    return sports['sport'].tolist(), models['model_name'].tolist()

# --- UI Layout -----------------------------------------------------------------------

st.title(f"🏈 ⚾️ {APP_TITLE} 🏀 🏒")

# --- User Inputs for Betting Strategy (MODIFIED) ---
st.markdown("### Betting Strategy Configuration")
c1, c2 = st.columns(2)
bankroll = c1.number_input("Enter your bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
# This is now a CAP, not a multiplier, to match TestModel's logic. Default is 1.0%
max_bet_percent = c2.number_input("Max Bet as % of Bankroll (The 'Safety Net')", min_value=0.1, max_value=100.0, value=1.0, step=0.1,
                                  help="Sets the MAXIMUM percentage of your bankroll to risk on any single bet, mirroring the 'max_fraction' in the backtest.")
max_bet_fraction = max_bet_percent / 100.0

# --- Sidebar for Game Filters ---
with st.sidebar:
    st.header("Filters")
    available_sports, available_models = get_filter_options()
    if not available_sports or not available_models:
        st.warning("No data in the database.")
    else:
        selected_sport = st.selectbox("Select Sport", available_sports)
        selected_date = st.date_input("Select Date", date.today())
        selected_model = st.selectbox("Select Model", available_models)

# --- Main Content Area ---
if 'selected_sport' in locals() and 'selected_model' in locals():
    df = load_data(selected_date, selected_sport, selected_model)
    if df.empty:
        st.warning(f"No games or predictions found for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}** with model **{selected_model}**.")
    else:
        st.success(f"Found {len(df)} games for **{selected_sport}** on **{selected_date.strftime('%Y-%m-%d')}**")
        for index, game in df.iterrows():
            with st.container(border=True):
                # Game Header
                col1, col2, col3 = st.columns([2.5, 1, 2.5])
                with col1:
                    st.image(game['team2_logo'], width=60)
                    st.subheader(f"{game['team2_name']} (Away)")
                with col2:
                    st.markdown("<h3 style='text-align: center; color: grey;'>VS</h3>", unsafe_allow_html=True)
                with col3:
                    st.image(game['team1_logo'], width=60)
                    st.subheader(f"{game['team1_name']} (Home)")
                st.divider()

                # --- Moneyline ---
                st.markdown("##### Moneyline")
                b1, b2 = st.columns([1.5, 2.5])
                with b1: # Bet Recommendation
                    prob_t1 = game.get('team1_win_prob')
                    odds_t1 = game.get('team1_moneyline')
                    dec_odds_t1 = american_to_decimal(odds_t1)
                    ev_t1 = calculate_ev(prob_t1, odds_t1)
                    kelly_t1 = calculate_kelly_fraction(prob_t1, dec_odds_t1)
                    prob_t2 = 1 - prob_t1 if prob_t1 is not None else None
                    odds_t2 = game.get('team2_moneyline')
                    dec_odds_t2 = american_to_decimal(odds_t2)
                    ev_t2 = calculate_ev(prob_t2, odds_t2)
                    kelly_t2 = calculate_kelly_fraction(prob_t2, dec_odds_t2)

                    if ev_t1 > 0 and ev_t1 > ev_t2:
                        bet_fraction = min(kelly_t1, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on {game['team1_name']} ({odds_t1:+.0f})")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t1*100:.2f}%")
                    elif ev_t2 > 0 and ev_t2 > ev_t1:
                        bet_fraction = min(kelly_t2, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on {game['team2_name']} ({odds_t2:+.0f})")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {ev_t2*100:.2f}%")
                    else:
                        st.info("No value found. Do not bet.")
                with b2: # Data Breakdown
                    imp_prob_t1 = american_to_prob(odds_t1)
                    imp_prob_t2 = american_to_prob(odds_t2)
                    st.dataframe({'Team': [game['team1_name'], game['team2_name']], 'Model Prob': [f"{prob_t1*100:.1f}%" if prob_t1 else 'N/A', f"{prob_t2*100:.1f}%" if prob_t2 else 'N/A'], 'Odds': [odds_t1, odds_t2], 'Implied Prob': [f"{imp_prob_t1*100:.1f}%" if imp_prob_t1 else 'N/A', f"{imp_prob_t2*100:.1f}%" if imp_prob_t2 else 'N/A'], 'EV': [f"{ev_t1*100:.2f}%", f"{ev_t2*100:.2f}%"]}, use_container_width=True)

                st.divider()
                
                # --- Spread ---
                st.markdown("##### Point Spread")
                sb1, sb2 = st.columns([1.5, 2.5])
                with sb1: # Bet Recommendation
                    prob_t1_cover = game.get('team1_cover_prob')
                    spread_t1 = game.get('team1_spread')
                    spread_odds_t1 = game.get('team1_spread_odds')
                    spread_dec_odds_t1 = american_to_decimal(spread_odds_t1)
                    spread_ev_t1 = calculate_ev(prob_t1_cover, spread_odds_t1)
                    spread_kelly_t1 = calculate_kelly_fraction(prob_t1_cover, spread_dec_odds_t1)
                    prob_t2_cover = 1 - prob_t1_cover if prob_t1_cover is not None else None
                    spread_t2 = -spread_t1 if spread_t1 is not None else None
                    spread_odds_t2 = game.get('team2_spread_odds')
                    spread_dec_odds_t2 = american_to_decimal(spread_odds_t2)
                    spread_ev_t2 = calculate_ev(prob_t2_cover, spread_odds_t2)
                    spread_kelly_t2 = calculate_kelly_fraction(prob_t2_cover, spread_dec_odds_t2)

                    if spread_ev_t1 > 0 and spread_ev_t1 > spread_ev_t2:
                        bet_fraction = min(spread_kelly_t1, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on {game['team1_name']} {spread_t1:+.1f}")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {spread_ev_t1*100:.2f}%")
                    elif spread_ev_t2 > 0 and spread_ev_t2 > spread_ev_t1:
                        bet_fraction = min(spread_kelly_t2, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on {game['team2_name']} {spread_t2:+.1f}")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {spread_ev_t2*100:.2f}%")
                    else:
                        st.info("No value found. Do not bet.")
                with sb2: # Data Breakdown
                    imp_prob_s1 = american_to_prob(spread_odds_t1)
                    imp_prob_s2 = american_to_prob(spread_odds_t2)
                    st.dataframe({'Bet': [f"{game['team1_name']} {spread_t1:+.1f}", f"{game['team2_name']} {spread_t2:+.1f}"], 'Model Prob': [f"{prob_t1_cover*100:.1f}%" if prob_t1_cover else 'N/A', f"{prob_t2_cover*100:.1f}%" if prob_t2_cover else 'N/A'], 'Odds': [spread_odds_t1, spread_odds_t2], 'Implied Prob': [f"{imp_prob_s1*100:.1f}%" if imp_prob_s1 else 'N/A', f"{imp_prob_s2*100:.1f}%" if imp_prob_s2 else 'N/A'], 'EV': [f"{spread_ev_t1*100:.2f}%", f"{spread_ev_t2*100:.2f}%"]}, use_container_width=True)

                st.divider()

                # --- Totals ---
                st.markdown("##### Total Score (Over/Under)")
                tb1, tb2 = st.columns([1.5, 2.5])
                with tb1:
                    prob_over = game.get('over_prob')
                    total_line = game.get('total_score')
                    over_odds = game.get('over_odds')
                    over_dec_odds = american_to_decimal(over_odds)
                    over_ev = calculate_ev(prob_over, over_odds)
                    over_kelly = calculate_kelly_fraction(prob_over, over_dec_odds)
                    prob_under = 1 - prob_over if prob_over is not None else None
                    under_odds = game.get('under_odds')
                    under_dec_odds = american_to_decimal(under_odds)
                    under_ev = calculate_ev(prob_under, under_odds)
                    under_kelly = calculate_kelly_fraction(prob_under, under_dec_odds)

                    if over_ev > 0 and over_ev > under_ev:
                        bet_fraction = min(over_kelly, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on Over {total_line:.1f}")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {over_ev*100:.2f}%")
                    elif under_ev > 0 and under_ev > over_ev:
                        bet_fraction = min(under_kelly, max_bet_fraction)
                        bet_size = bankroll * bet_fraction
                        st.success(f"✅ Bet on Under {total_line:.1f}")
                        st.metric("Suggested Wager", f"${bet_size:.2f}", f"Edge / EV: {under_ev*100:.2f}%")
                    else:
                        st.info("No value found. Do not bet.")
                with tb2:
                    imp_prob_o = american_to_prob(over_odds)
                    imp_prob_u = american_to_prob(under_odds)
                    st.dataframe({'Bet': [f"Over {total_line:.1f}", f"Under {total_line:.1f}"], 'Model Prob': [f"{prob_over*100:.1f}%" if prob_over else 'N/A', f"{prob_under*100:.1f}%" if prob_under else 'N/A'], 'Odds': [over_odds, under_odds], 'Implied Prob': [f"{imp_prob_o*100:.1f}%" if imp_prob_o else 'N/A', f"{imp_prob_u*100:.1f}%" if imp_prob_u else 'N/A'], 'EV': [f"{over_ev*100:.2f}%", f"{under_ev*100:.2f}%"]}, use_container_width=True)

        with st.expander("Show Raw Data Table"):
            st.dataframe(df)
else:
    st.info("Please select filters from the sidebar to view games.")