# team stats = teamrankings
# player stats = N/A (too difficult to get "as of" stats before a game)
# betting odds = ESPN API
# weather = some weather API
# final scores = ESPN API

import requests
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import difflib
import sqlite3
import pandas as pd
from io import StringIO


class Pregame:

    _ESPN_MAP = {
        'NBA': ('basketball', 'nba'),
        'NFL': ('football', 'nfl'),
        'CFB': ('football', 'college-football'),
        'CBB': ('basketball', 'mens-college-basketball'),
    }

    _TR_PREFIX = {
        'NBA': 'nba',
        'NFL': 'nfl',
        'CFB': 'college-football',
        'CBB': 'ncaa-basketball',
    }

        
    def __init__(self, date, sport: str, db_path: str = "sports.db"):
        self.sport = sport
        self.date = date
        self._tr_names = None

        # Open SQLite connection and ensure mapping table exists
        self.conn = sqlite3.connect(db_path)
        self._create_team_map_table()

    # MAIN METHOD TO BE CALLED
    def populate_pregame_data(self):
        pass

    # HELPER METHODS BELOW (Logic not implemented, placeholders only)

    def _create_team_map_table(self):
        """Create mapping table if it doesn't exist."""
        sql = """
        CREATE TABLE IF NOT EXISTS team_name_map (
            sport      TEXT NOT NULL,
            espn_name  TEXT NOT NULL,
            tr_name    TEXT NOT NULL,
            PRIMARY KEY (sport, espn_name)
        )
        """
        self.conn.execute(sql)
        self.conn.commit()

    def _load_teamrankings_names(self):
        prefix = self._TR_PREFIX[self.sport]
        url = (
            f"https://www.teamrankings.com/{prefix}/stat/points-per-game"
            f"?date={self.date.isoformat()}"
        )
        # if TeamRankings blocks non‐browser agents, add headers:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()

        # pandas will find the first <table> and return it as a DataFrame
        df = pd.read_html(StringIO(resp.text))[0]
        # the “Team” column holds exactly what you want:
        print(df["Team"].tolist())
        input()
        return df["Team"].tolist()
    
    def map_team_name(self, espn_name: str) -> str:
        # THIS WILL NOT WORK FOR NFL OR NBA TEAMS, SO THE NAMES HAVE JUST BEEN MANUALLY MAPPED IN THE DATABASE
        """
        1. Try to fetch tr_name from DB.
        2. If missing, fuzzy-match, insert into DB, then return.
        """
        # 1) Check DB
        cur = self.conn.execute(
            "SELECT tr_name FROM team_name_map WHERE sport=? AND espn_name=?",
            (self.sport, espn_name)
        )
        row = cur.fetchone()
        if row:
            return row[0]

        # 2) Not in DB: ensure list of TR names is loaded
        if self._tr_names is None:
            self._tr_names = self._load_teamrankings_names()

        # 3) Fuzzy-match
        matches = difflib.get_close_matches(espn_name, self._tr_names, n=1, cutoff=0.6)
        tr_name = matches[0] if matches else espn_name

        # 4) Insert new mapping
        self.conn.execute(
            "INSERT OR IGNORE INTO team_name_map (sport, espn_name, tr_name) VALUES (?, ?, ?)",
            (self.sport, espn_name, tr_name)
        )
        self.conn.commit()

        return tr_name

    def get_games_for_date(self):
        # 1) Build the URL from the sport mapping
        try:
            category, league = self._ESPN_MAP[self.sport]
        except KeyError:
            raise ValueError(f"Unsupported sport code: {self.sport!r}")

        url = (
            f"https://site.api.espn.com/apis/site/v2/"
            f"sports/{category}/{league}/scoreboard"
        )

        # 2) ESPN expects dates=YYYYMMDD
        date_str = self.date.strftime("%Y%m%d")
        resp = requests.get(url, params={"dates": date_str})
        resp.raise_for_status()
        data = resp.json()

        games = []
        for event in data.get("events", []):
            comp = event["competitions"][0]
            dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            days_since_epoch = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days

            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away = teams["away"]
            home = teams["home"]

            venue = comp.get("venue", {})
            addr  = venue.get("address", {})

            # season type
            season_type = event.get("season", {}).get("type")

            games.append({
                "id":                    event["id"],
                "date":                  event["date"],
                "days_since_epoch":      days_since_epoch,
                "game_time":             dt.time().isoformat(),

                "team1_id":              int(away["team"]["id"]),
                # THIS WILL NOT WORK FOR NFL OR NBA TEAMS, SO THE NAMES HAVE JUST BEEN MANUALLY MAPPED IN THE DATABASE
                "team1_name":            self.map_team_name(away["team"]["shortDisplayName"]),
                "team1_color":           away["team"].get("color"),
                "team1_alternate_color": away["team"].get("alternateColor"),
                "team1_logo":            away["team"].get("logo"),

                "team2_id":              int(home["team"]["id"]),
                # THIS WILL NOT WORK FOR NFL OR NBA TEAMS, SO THE NAMES HAVE JUST BEEN MANUALLY MAPPED IN THE DATABASE
                "team2_name":            self.map_team_name(home["team"]["shortDisplayName"]),
                "team2_color":           home["team"].get("color"),
                "team2_alternate_color": home["team"].get("alternateColor"),
                "team2_logo":            home["team"].get("logo"),

                "venue_id":              venue.get("id"),
                "city":                  addr.get("city"),
                "state":                 addr.get("state"),
                "country":               addr.get("country"),

                "is_neutral":            comp.get("neutralSite", False),
                "is_conference":         comp.get("conferenceCompetition", False),

                "season_type":           season_type,  # 1=pre,2=reg,3=post
            })

        return games
    
    def get_team_stats(self, team_name: str):
        """
        Query database or scrape stats for the team on specified date.
        Returns:
            Dict: { "stat_name": stat_value, ... }
        """
        pass

    def get_player_stats(self, team_id: int):
        """
        Query database or scrape stats for all players on the given team for the specified date.
        Returns:
            List[Dict]: [{"player_id": id, "stats": {stat_name: stat_value, ...}}, ...]
        """
        pass

    def get_betting_odds(self, game_id: str) -> dict:
        """
        Fetches moneyline, spread, and total (over/under) for the given ESPN event ID.
        Returns a dict:
            {
              'team1_moneyline': int or None,
              'team2_moneyline': int or None,
              'team1_spread': float or None,
              'team2_spread': float or None,
              'total_score': float or None
            }
        """
        category, league = self._ESPN_MAP[self.sport]
        url = (
            f"https://sports.core.api.espn.com/v2/sports/"
            f"{category}/leagues/{league}/events/{game_id}/"
            f"competitions/{game_id}/odds"
        )
        resp = requests.get(url)
        resp.raise_for_status()
        payload = resp.json()

        items = payload.get('items', [])
        if not items:
            return {
                'team1_moneyline': None,
                'team2_moneyline': None,
                'team1_spread':    None,
                'team2_spread':    None,
                'total_score':     None
            }

        # pick the provider with the highest priority
        entry = max(items, key=lambda e: e.get('provider', {}).get('priority', -999))

        away = entry.get('awayTeamOdds', {})
        home = entry.get('homeTeamOdds', {})

        # moneylines
        team1_ml = away.get('moneyLine')
        team2_ml = home.get('moneyLine')

        # spread: top-level 'spread' is a single float
        raw = entry.get('spread')
        if isinstance(raw, (int, float)):
            # if negative => home favoured by |raw|
            if raw < 0:
                team2_sp = raw
                team1_sp = -raw
            else:
                team1_sp = raw
                team2_sp = -raw
        else:
            # fallback to the juice if no top-level spread
            team1_sp = away.get('spreadOdds')
            team2_sp = home.get('spreadOdds')

        # total (over/under)
        total = entry.get('overUnder')

        return {
            'team1_moneyline': team1_ml,
            'team2_moneyline': team2_ml,
            'team1_spread':    team1_sp,
            'team2_spread':    team2_sp,
            'total_score':     total
        }

    def get_weather_info(self, location: str):
        """
        Fetch weather data given a location and date.
        Returns:
            Dict: {"temperature": val, "precipitation": val, "wind": val, ...}
        """
        pass

    def insert_pregame_row(self, game_id, team1_id, team2_id, stats, team2_moneyline, team2_spread, total_score):
        """
        Inserts the compiled pregame data into the pregame_data table.
        """
        pass

    def update_final_scores(self):
        """
        Scrape or call an API to fetch final scores for all games
        of self.sport on self.date, and write them back to the database.

        Workflow:
            1. Query DB for all game IDs on this date+sport.
            2. For each game ID:
            a. Fetch final scores (team1_score, team2_score).
            b. Upsert into your final_scores table (or update games table).
            3. Commit the transaction.
        """
        pass

    def __enter__(self):
        # Called at the start of a with‐block
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always close the DB, even if an exception occurred
        try:
            self.conn.close()
        except Exception:
            pass

        # Returning False lets any exception propagate
        return False