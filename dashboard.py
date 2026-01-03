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
import numpy as np
import plotly.express as px

# --- Path Correction & Model Imports --------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from Pregame import Pregame
    from TestModel import TestModel
except ImportError as e:
    st.error(f"FATAL: Could not import a required class. Ensure Pregame.py and TestModel.py are accessible.")
    st.exception(e) 
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
    ESPN_MAP = {
        'NBA': ('basketball', 'nba'), 
        'NFL': ('football', 'nfl'), 
        'CFB': ('football', 'college-football'), 
        'CBB': ('basketball', 'mens-college-basketball'), 
        'MLB': ('baseball', 'mlb')
    }
    
    if selected_sport not in ESPN_MAP: 
        return {}
    
    category, league = ESPN_MAP[selected_sport]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{category}/{league}/scoreboard"
    date_str = selected_date.strftime("%Y%m%d")
    
    try:
        params = {"dates": date_str}
        if selected_sport == "CBB":
            params.update({"groups": 50, "limit": 500})

        resp = requests.get(url, params=params, timeout=5)
        
        resp.raise_for_status()
        data = resp.json()
        
        for event in data.get("events", []):
            game_id = event.get("id")
            if not game_id: 
                continue
            
            comp = event["competitions"][0]
            status = comp.get("status", {}).get("type", {})
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away, home = teams.get("away"), teams.get("home")
            
            if away and home:
                live_data_map[game_id] = {
                    'away_score': int(away.get("score", 0)), 
                    'home_score': int(home.get("score", 0)), 
                    'status_detail': status.get("detail", "Scheduled")
                }
                
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

# --- Custom Audit Simulation Function ------------------------------------------------
def run_audit_simulation(analyzer, initial_bankroll=1000, max_fraction=0.01):
    """
    A specialized version of simulate_kelly_betting that logs every transaction
    into a DataFrame for the Audit Tab, including a BATCH ID.
    FIXED: Prevents intra-day compounding by sizing bets on 'start_of_day_bankroll'.
    """
    if not isinstance(analyzer.predictions, dict): return None
    
    # 1. Prepare Data
    o = analyzer._get_outcomes()
    probs = {'win': analyzer.predictions['win'][:, 1], 'spread': analyzer.predictions['spread'][:, 1], 'over': analyzer.predictions['over'][:, 1]}
    decimal_odds = {k: analyzer._american_to_decimal(v) for k, v in analyzer.test_odds.items()}
    dates = pd.to_datetime(analyzer.test_odds['date']) if 'date' in analyzer.test_odds.columns else pd.Series(range(len(analyzer.y_test)))
    
    sim_df = pd.DataFrame({
        'date': dates,
        'team1': analyzer.test_odds['team1_name'],
        'team2': analyzer.test_odds['team2_name'],
        'team1_score': analyzer.y_test['team1_score'] if isinstance(analyzer.y_test, pd.DataFrame) else analyzer.y_test[:,0],
        'team2_score': analyzer.y_test['team2_score'] if isinstance(analyzer.y_test, pd.DataFrame) else analyzer.y_test[:,1],
        'win_prob': probs['win'], 'spread_prob': probs['spread'], 'over_prob': probs['over'],
        'actual_winner_is_t1': o['actual_winner_is_t1'],
        'actual_spread_is_t1_cover': o['actual_spread_is_t1_cover'],
        'spread_pushes': o['spread_pushes'],
        'actual_is_over': o['actual_is_over'],
        'ou_pushes': o['ou_pushes'],
        'team1_ml': decimal_odds['team1_ml'], 'team2_ml': decimal_odds['team2_ml'],
        'team1_spread': decimal_odds['team1_spread_odds'], 'team2_spread': decimal_odds['team2_spread_odds'],
        'over_odds': decimal_odds['over_odds'], 'under_odds': decimal_odds['under_odds'],
        # Extra display info
        'team1_line': analyzer.test_odds['team1_spread'],
        'total_line': analyzer.test_odds['total_score']
    })

    sim_df = sim_df.sort_values('date')
    audit_logs = []

    # 2. Run Simulation for each Bet Type
    bet_types_config = {
        'Moneyline': {'prob': 'win_prob', 'o1': 'team1_ml', 'o2': 'team2_ml', 'outcome': 'actual_winner_is_t1', 'push': None},
        'Spread': {'prob': 'spread_prob', 'o1': 'team1_spread', 'o2': 'team2_spread', 'outcome': 'actual_spread_is_t1_cover', 'push': 'spread_pushes'},
        'Total': {'prob': 'over_prob', 'o1': 'over_odds', 'o2': 'under_odds', 'outcome': 'actual_is_over', 'push': 'ou_pushes'}
    }

    for bet_type_name, config in bet_types_config.items():
        bankroll = initial_bankroll
        batch_counter = 0 
        
        # --- DAILY BATCH LOOP ---
        for date, day_batch in sim_df.groupby('date'):
            batch_counter += 1
            daily_bets = [] 
            total_daily_fraction = 0.0
            
            # SNAPSHOT: Capture bankroll at the START of the day
            start_of_day_bankroll = bankroll 

            # PASS 1: Identify Bets
            for _, row in day_batch.iterrows():
                if config['push'] and row[config['push']]: continue
                
                odds1, odds2 = row[config['o1']], row[config['o2']]
                if np.isnan(odds1) or np.isnan(odds2): continue

                prob1 = row[config['prob']]
                k1 = calculate_kelly_fraction(prob1, odds1)
                k2 = calculate_kelly_fraction(1 - prob1, odds2)
                
                chosen_fraction = 0
                active_odds = 0
                won = False
                bet_description = ""

                if k1 > k2 and k1 > 0:
                    chosen_fraction = min(k1, max_fraction)
                    active_odds = odds1
                    won = (row[config['outcome']] == 1)
                    if bet_type_name == 'Moneyline': bet_description = f"{row['team1']} ML"
                    elif bet_type_name == 'Spread': bet_description = f"{row['team1']} {row['team1_line']}"
                    elif bet_type_name == 'Total': bet_description = f"Over {row['total_line']}"
                elif k2 > k1 and k2 > 0:
                    chosen_fraction = min(k2, max_fraction)
                    active_odds = odds2
                    won = (row[config['outcome']] == 0)
                    if bet_type_name == 'Moneyline': bet_description = f"{row['team2']} ML"
                    elif bet_type_name == 'Spread': bet_description = f"{row['team2']} {-row['team1_line']}"
                    elif bet_type_name == 'Total': bet_description = f"Under {row['total_line']}"
                
                if chosen_fraction > 0:
                    daily_bets.append({
                        'batch_id': batch_counter,
                        'date': date,
                        'game': f"{row['team1']} vs {row['team2']}",
                        'final_score': f"{int(row['team1_score'])}-{int(row['team2_score'])}",
                        'bet_type': bet_type_name,
                        'bet_on': bet_description,
                        'fraction': chosen_fraction,
                        'odds': active_odds,
                        'won': won
                    })
                    total_daily_fraction += chosen_fraction

            # SCALING: Resize bets if total fraction > 1.0
            scaling_factor = 1.0
            if total_daily_fraction > 1.0:
                scaling_factor = 1.0 / total_daily_fraction

            # PASS 2: Execute
            for bet in daily_bets:
                final_fraction = bet['fraction'] * scaling_factor
                
                # CRITICAL FIX: Calculate bet amount based on START OF DAY bankroll
                bet_amount = start_of_day_bankroll * final_fraction 
                
                pnl = 0
                if bet['won']:
                    profit = bet_amount * (bet['odds'] - 1)
                    pnl = profit
                    bankroll += profit # Update running bankroll for display
                else:
                    pnl = -bet_amount
                    bankroll -= bet_amount # Update running bankroll for display
                
                bet['wager'] = bet_amount
                bet['pnl'] = pnl
                bet['bankroll_after'] = bankroll
                bet['result'] = "WIN" if bet['won'] else "LOSS"
                audit_logs.append(bet)

    return pd.DataFrame(audit_logs)


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
c1, c2 = st.columns(2)
bankroll = c1.number_input("Enter your bankroll ($)", min_value=0.0, value=1000.0, step=100.0)
max_bet_percent = c2.number_input("Max Bet as % of Bankroll", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
max_bet_fraction = max_bet_percent / 100.0

tab1, tab2, tab3 = st.tabs(["Today's Betting Card", "Historical Performance", "Sanity Check (Audit)"])

# --- TAB 1: BETTING CARD (UNCHANGED) ---
with tab1:
    if 'selected_sport' in locals() and 'selected_model' in locals():
        df = load_data(selected_date, selected_sport, selected_model)
        live_scoreboard = fetch_live_scoreboard_data(selected_date, selected_sport)
        if df.empty:
            st.warning(f"No games found for **{selected_sport}** on **{selected_date}**.")
        else:
            state_key = f"{selected_sport}_{selected_date.strftime('%Y%m%d')}_{selected_model}"
            if 'bet_state_key' not in st.session_state or st.session_state.bet_state_key != state_key:
                st.session_state.bet_state_key = state_key
                ml_bets_list, spread_bets_list, total_bets_list = [], [], []
                for index, game in df.iterrows():
                    game_name = f"{game['team1_name']} @ {game['team2_name']}"
                    # Moneyline
                    prob_t1, odds_t1 = game.get('team1_win_prob'), game.get('team1_moneyline')
                    if calculate_ev(prob_t1, odds_t1) > 0: ml_bets_list.append({'Game': game_name, 'Bet Type': f"{game['team1_name']} ML", 'Line': 'N/A', 'Odds': odds_t1, 'Model Prob': prob_t1})
                    prob_t2 = 1 - prob_t1 if prob_t1 is not None else None
                    odds_t2 = game.get('team2_moneyline')
                    if calculate_ev(prob_t2, odds_t2) > 0: ml_bets_list.append({'Game': game_name, 'Bet Type': f"{game['team2_name']} ML", 'Line': 'N/A', 'Odds': odds_t2, 'Model Prob': prob_t2})
                    # Spread
                    prob_t1_cover, spread_t1, odds_t1_spread = game.get('team1_cover_prob'), game.get('team1_spread'), game.get('team1_spread_odds')
                    if calculate_ev(prob_t1_cover, odds_t1_spread) > 0: spread_bets_list.append({'Game': game_name, 'Bet Type': f"{game['team1_name']} {spread_t1:+.1f}", 'Line': spread_t1, 'Odds': odds_t1_spread, 'Model Prob': prob_t1_cover})
                    prob_t2_cover = 1 - prob_t1_cover if prob_t1_cover is not None else None
                    odds_t2_spread = game.get('team2_spread_odds')
                    if calculate_ev(prob_t2_cover, odds_t2_spread) > 0: spread_bets_list.append({'Game': game_name, 'Bet Type': f"{game['team2_name']} {-spread_t1:+.1f}", 'Line': -spread_t1, 'Odds': odds_t2_spread, 'Model Prob': prob_t2_cover})
                    # Totals
                    prob_over, total_line, over_odds = game.get('over_prob'), game.get('total_score'), game.get('over_odds')
                    if calculate_ev(prob_over, over_odds) > 0: total_bets_list.append({'Game': game_name, 'Bet Type': f"Over {total_line}", 'Line': total_line, 'Odds': over_odds, 'Model Prob': prob_over})
                    prob_under = 1 - prob_over if prob_over is not None else None
                    under_odds = game.get('under_odds')
                    if calculate_ev(prob_under, under_odds) > 0: total_bets_list.append({'Game': game_name, 'Bet Type': f"Under {total_line}", 'Line': total_line, 'Odds': under_odds, 'Model Prob': prob_under})

                st.session_state.ml_df = pd.DataFrame(ml_bets_list) if ml_bets_list else pd.DataFrame()
                st.session_state.spread_df = pd.DataFrame(spread_bets_list) if spread_bets_list else pd.DataFrame()
                st.session_state.total_df = pd.DataFrame(total_bets_list) if total_bets_list else pd.DataFrame()

            def process_and_display_bets(title, df_key, editor_key):
                st.markdown(f"#### {title}")
                if df_key not in st.session_state or st.session_state[df_key].empty:
                    st.info(f"No positive EV {title.lower()} found.")
                    return
                temp_df = st.session_state[df_key].copy()
                temp_df['EV'] = temp_df.apply(lambda row: calculate_ev(row['Model Prob'], row['Odds']) * 100, axis=1)
                def calc_kelly_display(row):
                    dec_odds = american_to_decimal(row['Odds'])
                    kelly_frac = calculate_kelly_fraction(row['Model Prob'], dec_odds)
                    bet_size = bankroll * min(kelly_frac, max_bet_fraction)
                    return f"${bet_size:.2f}"
                temp_df['Kelly Bet'] = temp_df.apply(calc_kelly_display, axis=1)
                edited_df = st.data_editor(temp_df, key=editor_key, use_container_width=True, hide_index=True, disabled=["Game", "Bet Type", "Line", "Model Prob", "EV", "Kelly Bet"])
                if not edited_df['Odds'].equals(temp_df['Odds']):
                    st.session_state[df_key]['Odds'] = edited_df['Odds']
                    st.rerun()

            st.markdown("### Recommended Bets Summary")
            process_and_display_bets("Moneyline Bets", 'ml_df', 'ml_editor')
            process_and_display_bets("Spread Bets", 'spread_df', 'spread_editor')
            process_and_display_bets("Over/Under Bets", 'total_df', 'total_editor')
            st.divider()

            st.success(f"Found {len(df)} games for **{selected_sport}** on **{selected_date}**")
            st.markdown("### Game-by-Game Breakdown")
            for index, game in df.iterrows():
                with st.container(border=True):
                    col1, col2, col3 = st.columns([2.5, 1.5, 2.5])
                    with col1:
                        if pd.notna(game['team1_logo']) and game['team1_logo']: st.image(game['team1_logo'], width=60)
                        st.subheader(f"{game['team1_name']}")
                    with col2:
                        live_data = live_scoreboard.get(str(game['game_id']), {})
                        st.metric(label=live_data.get('status_detail', 'Scheduled'), value=f"{live_data.get('away_score',0)} – {live_data.get('home_score',0)}")
                    with col3:
                        if pd.notna(game['team2_logo']) and game['team2_logo']: st.image(game['team2_logo'], width=60)
                        st.subheader(f"{game['team2_name']}")
    else:
        st.info("Select filters.")

# --- TAB 2: HISTORICAL (UNCHANGED) ---
with tab2:
    st.header(f"Historical Performance Review")
    if 'selected_sport' in locals() and 'selected_model' in locals():
        min_hist_date, max_hist_date = get_historical_date_range(selected_sport)
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Start Date", min_hist_date, min_value=min_hist_date, max_value=max_hist_date)
        end_date = c2.date_input("End Date", max_hist_date, min_value=min_hist_date, max_value=max_hist_date)

        if start_date <= end_date:
            st.markdown("---")
            with st.spinner(f"Running historical analysis..."):
                hist_df = load_historical_data_for_testmodel(selected_sport, selected_model, start_date, end_date)
                if not hist_df.empty:
                    y_test = hist_df[['team1_score', 'team2_score']].to_numpy()
                    hist_df.rename(columns={'team1_moneyline': 'team1_ml', 'team2_moneyline': 'team2_ml'}, inplace=True, errors='ignore')
                    predictions = {'win': hist_df[['team1_win_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(), 'spread': hist_df[['team1_cover_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list(), 'over': hist_df[['over_prob']].apply(lambda x: [1-x.iloc[0], x.iloc[0]], axis=1).to_list()}
                    for key in predictions: predictions[key] = pd.DataFrame(predictions[key]).to_numpy()

                    analyzer = TestModel(predictions=predictions, y_test=y_test, test_odds=hist_df, dates=hist_df['date'])
                    
                    text_output = io.StringIO()
                    with redirect_stdout(text_output):
                        analyzer.display_results(initial_bankroll=bankroll, max_fraction=max_bet_fraction) # Will use print statements
                    st.code(text_output.getvalue(), language='text')

                    if isinstance(analyzer.predictions, dict):
                        plotting_functions = {
                            "Model vs. Market Calibration": analyzer.check_calibration,
                            "Monte Carlo": lambda: analyzer.run_probabilistic_monte_carlo(initial_bankroll=bankroll, max_fraction=max_bet_fraction),
                        }
                        for title, func in plotting_functions.items():
                            st.subheader(title)
                            plot_output = io.StringIO()
                            with redirect_stdout(plot_output):
                                plt.close('all')
                                func()
                                fig = plt.gcf()
                            if fig.get_axes(): st.pyplot(fig)
                            plt.close(fig)
                else:
                    st.warning("No historical data found.")

# --- TAB 3: SANITY CHECK (UPDATED) ---
with tab3:
    st.header("🕵️‍♂️ Sanity Check (Audit Log)")
    st.markdown("Verify every single bet the simulation makes. Check for 'Time Travel' errors (betting on games with known scores) or logic errors.")
    
    if 'analyzer' in locals() and 'hist_df' in locals() and not hist_df.empty:
        st.info(f"Auditing simulation from {start_date} to {end_date}...")
        
        audit_df = run_audit_simulation(analyzer, initial_bankroll=bankroll, max_fraction=max_bet_fraction)
        
        if audit_df is not None and not audit_df.empty:
            # Filters for the audit table
            bet_type_filter = st.multiselect("Filter by Bet Type", audit_df['bet_type'].unique(), default=audit_df['bet_type'].unique())
            result_filter = st.multiselect("Filter by Result", ['WIN', 'LOSS'], default=['WIN', 'LOSS'])
            
            filtered_audit = audit_df[
                (audit_df['bet_type'].isin(bet_type_filter)) & 
                (audit_df['result'].isin(result_filter))
            ]
            
            st.metric("Total Bets Placed", len(filtered_audit))
            
            # Formatting and Display
            st.dataframe(
                filtered_audit.style.format({
                    'batch_id': 'Batch {}',
                    'fraction': '{:.2%}',
                    'odds': '{:.2f}',
                    'wager': '${:.2f}',
                    'pnl': '${:+.2f}',
                    'bankroll_after': '${:.2f}',
                    'date': '{:%Y-%m-%d}'
                }).map(lambda x: 'color: green' if x == 'WIN' else 'color: red', subset=['result']),
                use_container_width=True,
                column_order=['batch_id', 'date', 'game', 'final_score', 'bet_type', 'bet_on', 'result', 'wager', 'pnl', 'bankroll_after']
            )
            
            # Plot bankroll over time for sanity check
            st.subheader("Bankroll Evolution (Sanity Check)")
            # Group by date to show end-of-day bankroll
            daily_end_bankroll = audit_df.groupby(['date', 'bet_type'])['bankroll_after'].last().unstack()
            fig = px.line(daily_end_bankroll, markers=True)
            fig.update_traces(connectgaps=True)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("No bets were placed during this period (or data is missing).")
    else:
        st.info("⚠️ Please run the **Historical Performance** tab first to load the data for the audit.")