# team stats = teamrankings
# player stats = N/A (too difficult to get "as of" stats before a game)
# betting odds = ESPN API
# weather = some weather API
# final scores = ESPN API

import requests
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import difflib
import sqlite3
import pandas as pd
from io import StringIO
import logging
import re
from urllib.parse import urlparse
import json
import time
from requests.exceptions import HTTPError

class Pregame:

    _ESPN_MAP = {
        'NBA': ('basketball', 'nba'),
        'NFL': ('football', 'nfl'),
        'CFB': ('football', 'college-football'),
        'CBB': ('basketball', 'mens-college-basketball'),
        'MLB': ('baseball', 'mlb'),
    }

    _TR_PREFIX = {
        'NBA': 'nba',
        'NFL': 'nfl',
        'CFB': 'college-football',
        'CBB': 'ncaa-basketball',
        'MLB': 'mlb',
    }

        
    def __init__(self, date, sport: str, db_path: str = "sports.db"):
        self.sport = sport
        self.date = date
        self._tr_names = None
        self._stats_cache = {}
        self._stats_cache_valid = True

        # Open SQLite connection and ensure mapping table exists
        self.conn = sqlite3.connect(db_path)
        self._create_team_map_table()

    # MAIN METHOD TO BE CALLED
    def populate_pregame_data(self):
        """
        For each game on self.date/self.sport:
          - Fetch team1/team2 raw & normalized stats
          - Fetch odds & weather
          - Build two JSON blobs (raw_stats, normalized_stats)
          - INSERT OR REPLACE into games table
        """
        games = self.get_games_for_date()
        inserted = 0
        skip = 0

        for g in games:
            # -- validate/convert venue_id --
            vid = g.get("venue_id")
            if vid is not None:
                try:
                    venue_int = int(vid)
                except (TypeError, ValueError):
                    logging.error(f"Bad venue_id {vid!r}, skipping game {g['id']}")
                    skip += 1
                    continue
            else:
                venue_int = None

            # -- pull team stats --
            t1 = self.get_team_stats(g["team1_name"])
            t2 = self.get_team_stats(g["team2_name"])
            if t1 is None or t2 is None:
                skip += 1
                logging.error(f"Could not fetch stats for {g['team1_name']} or {g['team2_name']}, skipping")
                continue

            # split out raw vs normalized
            raw_t1 = {slug: info["raw"] for slug, info in t1.items()}
            raw_t2 = {slug: info["raw"] for slug, info in t2.items()}
            norm_t1 = {slug: info["normalized"] for slug, info in t1.items()}
            norm_t2 = {slug: info["normalized"] for slug, info in t2.items()}

            # -- fetch weather --
            weather = self.get_weather_info(
                time_iso=g["game_time"] + "Z",
                city=g["city"],
                state=g["state"],
                country=g["country"]
            )
            if not weather:
                logging.error(f"Could not fetch weather for game {g['id']} at {g['city']}, skipping")
                skip += 1
                continue

            # -- fetch odds --
            odds = self.get_betting_odds(g["id"])

            # -- compute date parts & day-of-week & float time --
            dt = datetime.fromisoformat(g["date"].replace("Z", "+00:00"))
            day, month, year = dt.day, dt.month, dt.year
            dow = g["day_of_the_week"]  # Monday=0…Sunday=6
            time_float = dt.hour + dt.minute/60 + dt.second/3600

            # -- assemble JSON blobs --
            stats_blob = {
                "day": day,
                "month": month,
                "year": year,
                "days_since_epoch": g["days_since_epoch"],
                "game_time": time_float,
                "day_of_week": dow,
                "team1_id": g["team1_id"],
                "team2_id": g["team2_id"],
                "venue_id": venue_int,
                "is_neutral": g["is_neutral"],
                "is_conference": g["is_conference"],
                "season_type": g.get("season_type"),
                "team1_stats": raw_t1,
                "team2_stats": raw_t2,
                "weather": weather
            }

            normalized_blob = {
                "day": day,
                "month": month,
                "year": year,
                "days_since_epoch": g["days_since_epoch"],
                "game_time": time_float,
                "day_of_week": dow,
                "team1_id": g["team1_id"],
                "team2_id": g["team2_id"],
                "venue_id": venue_int,
                "is_neutral": g["is_neutral"],
                "is_conference": g["is_conference"],
                "season_type": g.get("season_type"),
                "team1_stats": norm_t1,
                "team2_stats": norm_t2,
                "weather": weather
            }

            # -- upsert into SQLite --
            self.conn.execute("""
                INSERT OR REPLACE INTO games (
                    game_id, date, days_since_epoch, day_of_the_week, game_time, sport,
                    team1_id, team1_name, team1_color, team1_alt_color, team1_logo,
                    team2_id, team2_name, team2_color, team2_alt_color, team2_logo,
                    venue_id, city, state, country, is_neutral, is_conference, season_type,
                    stats, normalized_stats,
                    team1_moneyline, team2_moneyline,
                    team1_spread,  team2_spread,
                    team1_spread_odds, team2_spread_odds,
                    total_score, over_odds, under_odds
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
                )
            """, (
                g["id"],
                self.date.isoformat(),
                g["days_since_epoch"],
                dow,
                g["game_time"],
                self.sport,

                g["team1_id"],
                g["team1_name"],
                g["team1_color"],
                g["team1_alternate_color"],
                g["team1_logo"],

                g["team2_id"],
                g["team2_name"],
                g["team2_color"],
                g["team2_alternate_color"],
                g["team2_logo"],

                venue_int,
                g["city"],
                g["state"],
                g["country"],
                int(g["is_neutral"]),
                int(g["is_conference"]),
                g.get("season_type"),

                json.dumps(stats_blob),
                json.dumps(normalized_blob),

                # existing odds:
                odds.get("team1_moneyline"),
                odds.get("team2_moneyline"),
                odds.get("team1_spread"),
                odds.get("team2_spread"),
                # new spread‐odds fields:
                odds.get("team1_spread_odds"),
                odds.get("team2_spread_odds"),
                # existing total and new over/under odds:
                odds.get("total_score"),
                odds.get("over_odds"),
                odds.get("under_odds"),
            )
            )
            inserted += 1

        self.conn.commit()
        logging.info(f"populate_pregame_data: inserted/updated {inserted} rows, skipped {skip} games.")
        return skip

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
            f"https://www.teamrankings.com/{prefix}/ranking/predictive-by-other/"
            f"?date={self.date.isoformat()}"
        )
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/114.0.0.0 Safari/537.36"
            )
        }

        while True:
            resp = requests.get(url, headers=headers)
            try:
                resp.raise_for_status()
                break
            except HTTPError as e:
                # if blocked, wait 4 minutes and retry
                if resp.status_code == 403:
                    logging.warning(f"403 Forbidden at {url}, sleeping 4 minutes before retry")
                    time.sleep(4 * 60)
                    continue
                # other HTTP errors bubble up
                raise

        df = pd.read_html(StringIO(resp.text))[0]
        teams = df["Team"].tolist()

        # strip only trailing "(W-L)" (e.g. "(14-2)"), leave other parentheses intact
        cleaned = [
            re.sub(r"\s*\(\d+-\d+\)\s*$", "", name)
            for name in teams
        ]
        return cleaned
    
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
        # for CBB (men's college basketball), request all teams, not just Top-25
        params = {"dates": date_str}
        if self.sport == "CBB":
            params.update({"groups": 50, "limit": 500})
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        now_utc = datetime.now(timezone.utc)
        games = []

        for event in data.get("events", []):
            comp = event["competitions"][0]
            # original UTC datetime from ESPN
            dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))

            # if the game is in the past, subtract local offset
            if dt < now_utc - timedelta(hours=16):
                if self.sport in ("CFB", "NFL"):
                    dt -= timedelta(hours=5, minutes=30)
                elif self.sport in ("CBB", "NBA", "MLB"):
                    dt -= timedelta(hours=4, minutes=30)
                # otherwise no adjustment

            # recompute these from the (possibly adjusted) dt
            days_since_epoch = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days
            day_of_week = dt.weekday()  # Monday=0…Sunday=6

            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away = teams["away"]
            home = teams["home"]

            venue = comp.get("venue", {})
            addr  = venue.get("address", {})

            season_type = event.get("season", {}).get("type")

            games.append({
                "id":                    event["id"],
                "date":                  dt.isoformat(),  # original ISO timestamp
                "days_since_epoch":      days_since_epoch,
                "day_of_the_week":       day_of_week,
                "game_time":             dt.time().isoformat(),

                "team1_id":              int(away["team"]["id"]),
                "team1_name":            self.map_team_name(away["team"]["shortDisplayName"]),
                "team1_color":           away["team"].get("color"),
                "team1_alternate_color": away["team"].get("alternateColor"),
                "team1_logo":            away["team"].get("logo"),

                "team2_id":              int(home["team"]["id"]),
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
    
    def _parse_stat_value(self, text: str) -> float:
        """
        Turn a table cell into a single float:
         - "--"           → 0.0
         - "W-L"          → win_pct = W/(W+L)
         - "MM:SS"        → minutes + seconds/60
         - "42.3"         → 42.3
         - "75%"          → 0.75
         - otherwise log and return 0.0
        """
        text = text.strip()
        if text == "--":
            return 0.0

        # ratio form "2-1"
        if re.fullmatch(r"\d+-\d+", text):
            w, l = map(int, text.split("-"))
            total = w + l
            return (w / total) if total else 0.0
        
        # time form "MM:SS" → minutes as float
        if re.fullmatch(r"\d+:\d{2}", text):
            m, s = text.split(":")
            try:
                m, s = int(m), int(s)
                return m + s/60
            except ValueError:
                logging.error(f"Cannot parse time value: {text!r}")
                return 0.0

        # percent form
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100
            except ValueError:
                logging.error(f"Cannot parse percent value: {text!r}")
                return 0.0

        # plain number
        try:
            return float(text)
        except ValueError:
            logging.error(f"Unrecognized stat value: {text}")
            return 0.0

    def get_team_stats(self, team_name: str):
        """
        For this.sport on this.date, scrape each URL only once into a cache,
        then for each call extract raw & normalized stats for team_name.

        Returns:
            Dict[str, Dict[str, Dict[str, float]]]:
            {
                "<stat-slug>": {
                "raw":        { "Rank": 5.0, "2024": 28.9, … },
                "normalized": { "Rank": 0.04, "2024": 0.83, … }
                },
                …
            }
        If a table is missing, raises RuntimeError.
        If team_name is not found on any page, logs and returns None.
        """

        if not self._stats_cache_valid:
            return None
        
        # define once
        url_lists = {
            'CBB': [
                "https://www.teamrankings.com/ncaa-basketball/stat/points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/average-scoring-margin",
                "https://www.teamrankings.com/ncaa-basketball/stat/offensive-efficiency",
                "https://www.teamrankings.com/ncaa-basketball/stat/floor-percentage",
                "https://www.teamrankings.com/ncaa-basketball/stat/1st-half-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/2nd-half-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/overtime-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/average-1st-half-margin",
                "https://www.teamrankings.com/ncaa-basketball/stat/average-2nd-half-margin",
                "https://www.teamrankings.com/ncaa-basketball/stat/average-overtime-margin",
                "https://www.teamrankings.com/ncaa-basketball/stat/points-from-2-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/points-from-3-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-2-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-3-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/percent-of-points-from-free-throws",
                "https://www.teamrankings.com/ncaa-basketball/stat/shooting-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/effective-field-goal-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/three-point-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/two-point-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/free-throw-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/true-shooting-percentage",
                "https://www.teamrankings.com/ncaa-basketball/stat/field-goals-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/field-goals-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/three-pointers-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/three-pointers-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/free-throws-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/free-throws-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/three-point-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/two-point-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/fta-per-fga",
                "https://www.teamrankings.com/ncaa-basketball/stat/ftm-per-100-possessions",
                "https://www.teamrankings.com/ncaa-basketball/stat/free-throw-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/non-blocked-2-pt-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/offensive-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/defensive-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/team-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/total-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/offensive-rebounding-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/defensive-rebounding-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/total-rebounding-percentage",
                "https://www.teamrankings.com/ncaa-basketball/stat/blocks-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/steals-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/block-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/steals-perpossession",
                "https://www.teamrankings.com/ncaa-basketball/stat/steal-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/turnovers-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/turnovers-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/assist--per--turnover-ratio",
                "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-fgm",
                "https://www.teamrankings.com/ncaa-basketball/stat/assists-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/turnover-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/personal-fouls-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/personal-fouls-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/personal-foul-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-average-scoring-margin",
                "https://www.teamrankings.com/ncaa-basketball/stat/defensive-efficiency",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-floor-percentage",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-1st-half-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-2nd-half-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-overtime-points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-from-2-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-points-from-3-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-2-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-3-pointers",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-percent-of-points-from-free-throws",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-shooting-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-effective-field-goal-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-point-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-two-point-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throw-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-true-shooting-percentage",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-field-goals-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-field-goals-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-pointers-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-pointers-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throws-made-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throws-attempted-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-three-point-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-two-point-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-fta-per-fga",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-ftm-per-100-possessions",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-free-throw-rate",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-non-blocked-2-pt-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-offensive-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-defensive-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-team-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-total-rebounds-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-offensive-rebounding-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-defensive-rebounding-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-blocks-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steals-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-block-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steals-perpossession",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-steal-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnovers-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assist--per--turnover-ratio",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-fgm",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-assists-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnovers-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-turnover-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-fouls-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-fouls-per-possession",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-personal-foul-pct",
                "https://www.teamrankings.com/ncaa-basketball/stat/games-played",
                "https://www.teamrankings.com/ncaa-basketball/stat/possessions-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/extra-chances-per-game",
                "https://www.teamrankings.com/ncaa-basketball/stat/effective-possession-ratio",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-effective-possession-ratio",
                "https://www.teamrankings.com/ncaa-basketball/stat/win-pct-all-games",
                "https://www.teamrankings.com/ncaa-basketball/stat/win-pct-close-games",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-win-pct-all-games",
                "https://www.teamrankings.com/ncaa-basketball/stat/opponent-win-pct-close-games",
                "https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/home-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/away-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/neutral-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/home-adv-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/schedule-strength-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/future-sos-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/season-sos-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/sos-basic-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/in-conference-sos-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/non-conference-sos-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/last-5-games-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/last-10-games-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/in-conference-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/non-conference-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/luck-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/consistency-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/vs-1-25-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/vs-26-50-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/vs-51-100-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/vs-101-200-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/vs-201-and-up-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/first-half-by-other",
                "https://www.teamrankings.com/ncaa-basketball/ranking/second-half-by-other",
                "https://www.teamrankings.com/ncaa-basketball/rpi-ranking/rpi-rating-by-team",
                "https://www.teamrankings.com/ncaa-basketball/rpi-ranking/sos-rpi-rating-by-team"
                ],
            'CFB': [
                "https://www.teamrankings.com/college-football/stat/points-per-game",
                "https://www.teamrankings.com/college-football/stat/average-scoring-margin",
                "https://www.teamrankings.com/college-football/stat/yards-per-point",
                "https://www.teamrankings.com/college-football/stat/yards-per-point-margin",
                "https://www.teamrankings.com/college-football/stat/points-per-play",
                "https://www.teamrankings.com/college-football/stat/points-per-play-margin",
                "https://www.teamrankings.com/college-football/stat/red-zone-scoring-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/red-zone-scores-per-game",
                "https://www.teamrankings.com/college-football/stat/red-zone-scoring-pct",
                "https://www.teamrankings.com/college-football/stat/offensive-touchdowns-per-game",
                "https://www.teamrankings.com/college-football/stat/offensive-points-per-game",
                "https://www.teamrankings.com/college-football/stat/offensive-point-share-pct",
                "https://www.teamrankings.com/college-football/stat/1st-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/2nd-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/3rd-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/4th-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/overtime-points-per-game",
                "https://www.teamrankings.com/college-football/stat/1st-half-points-per-game",
                "https://www.teamrankings.com/college-football/stat/2nd-half-points-per-game",
                "https://www.teamrankings.com/college-football/stat/1st-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/2nd-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/3rd-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/4th-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/1st-half-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/2nd-half-time-of-possession-share-pct",
                "https://www.teamrankings.com/college-football/stat/yards-per-game",
                "https://www.teamrankings.com/college-football/stat/plays-per-game",
                "https://www.teamrankings.com/college-football/stat/yards-per-play",
                "https://www.teamrankings.com/college-football/stat/third-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/third-down-conversions-per-game",
                "https://www.teamrankings.com/college-football/stat/fourth-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/fourth-down-conversions-per-game",
                "https://www.teamrankings.com/college-football/stat/average-time-of-possession-net-of-ot",
                "https://www.teamrankings.com/college-football/stat/time-of-possession-pct-net-of-ot",
                "https://www.teamrankings.com/college-football/stat/seconds-per-play",
                "https://www.teamrankings.com/college-football/stat/third-down-conversion-pct",
                "https://www.teamrankings.com/college-football/stat/fourth-down-conversion-pct",
                "https://www.teamrankings.com/college-football/stat/punts-per-play",
                "https://www.teamrankings.com/college-football/stat/punts-per-offensive-score",
                "https://www.teamrankings.com/college-football/stat/rushing-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/rushing-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/yards-per-rush-attempt",
                "https://www.teamrankings.com/college-football/stat/rushing-play-pct",
                "https://www.teamrankings.com/college-football/stat/rushing-yards-pct",
                "https://www.teamrankings.com/college-football/stat/pass-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/completions-per-game",
                "https://www.teamrankings.com/college-football/stat/incompletions-per-game",
                "https://www.teamrankings.com/college-football/stat/completion-pct",
                "https://www.teamrankings.com/college-football/stat/passing-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/qb-sacked-per-game",
                "https://www.teamrankings.com/college-football/stat/qb-sacked-pct",
                "https://www.teamrankings.com/college-football/stat/average-team-passer-rating",
                "https://www.teamrankings.com/college-football/stat/passing-play-pct",
                "https://www.teamrankings.com/college-football/stat/passing-yards-pct",
                "https://www.teamrankings.com/college-football/stat/yards-per-pass-attempt",
                "https://www.teamrankings.com/college-football/stat/yards-per-completion",
                "https://www.teamrankings.com/college-football/stat/field-goal-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/field-goals-made-per-game",
                "https://www.teamrankings.com/college-football/stat/field-goal-conversion-pct",
                "https://www.teamrankings.com/college-football/stat/punt-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/gross-punt-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-yards-per-point",
                "https://www.teamrankings.com/college-football/stat/opponent-points-per-play",
                "https://www.teamrankings.com/college-football/stat/opponent-average-scoring-margin",
                "https://www.teamrankings.com/college-football/stat/opponent-red-zone-scoring-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-red-zone-scores-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-red-zone-scoring-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-points-per-field-goal-attempt",
                "https://www.teamrankings.com/college-football/stat/opponent-offensive-touchdowns-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-offensive-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-1st-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-2nd-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-3rd-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-4th-quarter-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opp-overtime-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-1st-half-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-2nd-half-points-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-plays-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-yards-per-play",
                "https://www.teamrankings.com/college-football/stat/opponent-first-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-third-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-third-down-conversions-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-fourth-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-fourth-down-conversions-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-average-time-of-possession-net-of-ot",
                "https://www.teamrankings.com/college-football/stat/opponent-time-of-possession-pct-net-of-ot",
                "https://www.teamrankings.com/college-football/stat/opponent-seconds-per-play",
                "https://www.teamrankings.com/college-football/stat/opponent-third-down-conversion-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-fourth-down-conversion-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-punts-per-play",
                "https://www.teamrankings.com/college-football/stat/opponent-punts-per-offensive-score",
                "https://www.teamrankings.com/college-football/stat/opponent-rushing-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-rushing-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-rushing-first-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-yards-per-rush-attempt",
                "https://www.teamrankings.com/college-football/stat/opponent-rushing-play-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-rushing-yards-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-pass-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-completions-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-incompletions-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-completion-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-passing-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-passing-first-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-average-team-passer-rating",
                "https://www.teamrankings.com/college-football/stat/sack-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-passing-play-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-passing-yards-pct",
                "https://www.teamrankings.com/college-football/stat/sacks-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-yards-per-pass-attempt",
                "https://www.teamrankings.com/college-football/stat/opponent-yards-per-completion",
                "https://www.teamrankings.com/college-football/stat/opponent-field-goal-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-field-goals-made-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-punt-attempts-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-gross-punt-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/interceptions-thrown-per-game",
                "https://www.teamrankings.com/college-football/stat/fumbles-per-game",
                "https://www.teamrankings.com/college-football/stat/fumbles-lost-per-game",
                "https://www.teamrankings.com/college-football/stat/fumbles-not-lost-per-game",
                "https://www.teamrankings.com/college-football/stat/giveaways-per-game",
                "https://www.teamrankings.com/college-football/stat/turnover-margin-per-game",
                "https://www.teamrankings.com/college-football/stat/interceptions-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-fumbles-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-fumbles-lost-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-fumbles-not-lost-per-game",
                "https://www.teamrankings.com/college-football/stat/takeaways-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-turnover-margin-per-game",
                "https://www.teamrankings.com/college-football/stat/pass-intercepted-pct",
                "https://www.teamrankings.com/college-football/stat/fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/giveaway-fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/takeaway-fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/interception-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-giveaway-fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/opponent-takeaway-fumble-recovery-pct",
                "https://www.teamrankings.com/college-football/stat/penalties-per-game",
                "https://www.teamrankings.com/college-football/stat/penalty-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/penalty-first-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-penalties-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-penalty-yards-per-game",
                "https://www.teamrankings.com/college-football/stat/opponent-penalty-first-downs-per-game",
                "https://www.teamrankings.com/college-football/stat/penalty-yards-per-penalty",
                "https://www.teamrankings.com/college-football/stat/penalties-per-play",
                "https://www.teamrankings.com/college-football/stat/opponent-penalty-yards-per-penalty",
                "https://www.teamrankings.com/college-football/stat/opponent-penalties-per-play",
                "https://www.teamrankings.com/college-football/ranking/predictive-by-other",
                "https://www.teamrankings.com/college-football/ranking/home-by-other",
                "https://www.teamrankings.com/college-football/ranking/away-by-other",
                "https://www.teamrankings.com/college-football/ranking/neutral-by-other",
                "https://www.teamrankings.com/college-football/ranking/home-adv-by-other",
                "https://www.teamrankings.com/college-football/ranking/schedule-strength-by-other",
                "https://www.teamrankings.com/college-football/ranking/future-sos-by-other",
                "https://www.teamrankings.com/college-football/ranking/season-sos-by-other",
                "https://www.teamrankings.com/college-football/ranking/sos-basic-by-other",
                "https://www.teamrankings.com/college-football/ranking/in-conference-sos-by-other",
                "https://www.teamrankings.com/college-football/ranking/non-conference-sos-by-other",
                "https://www.teamrankings.com/college-football/ranking/last-5-games-by-other",
                "https://www.teamrankings.com/college-football/ranking/last-10-games-by-other",
                "https://www.teamrankings.com/college-football/ranking/in-conference-by-other",
                "https://www.teamrankings.com/college-football/ranking/non-conference-by-other",
                "https://www.teamrankings.com/college-football/ranking/luck-by-other",
                "https://www.teamrankings.com/college-football/ranking/consistency-by-other",
                "https://www.teamrankings.com/college-football/ranking/vs-1-10-by-other",
                "https://www.teamrankings.com/college-football/ranking/vs-11-25-by-other",
                "https://www.teamrankings.com/college-football/ranking/vs-26-40-by-other",
                "https://www.teamrankings.com/college-football/ranking/vs-41-75-by-other",
                "https://www.teamrankings.com/college-football/ranking/vs-76-120-by-other",
                "https://www.teamrankings.com/college-football/ranking/first-half-by-other",
                "https://www.teamrankings.com/college-football/ranking/second-half-by-other"
                ],
            'NBA': [
                "https://www.teamrankings.com/nba/stat/points-per-game",
                "https://www.teamrankings.com/nba/stat/average-scoring-margin",
                "https://www.teamrankings.com/nba/stat/offensive-efficiency",
                "https://www.teamrankings.com/nba/stat/floor-percentage",
                "https://www.teamrankings.com/nba/stat/1st-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/2nd-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/3rd-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/4th-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/1st-half-points-per-game",
                "https://www.teamrankings.com/nba/stat/2nd-half-points-per-game",
                "https://www.teamrankings.com/nba/stat/overtime-points-per-game",
                "https://www.teamrankings.com/nba/stat/points-in-paint-per-game",
                "https://www.teamrankings.com/nba/stat/fastbreak-points-per-game",
                "https://www.teamrankings.com/nba/stat/fastbreak-efficiency",
                "https://www.teamrankings.com/nba/stat/average-biggest-lead",
                "https://www.teamrankings.com/nba/stat/average-1st-quarter-margin",
                "https://www.teamrankings.com/nba/stat/average-2nd-quarter-margin",
                "https://www.teamrankings.com/nba/stat/average-3rd-quarter-margin",
                "https://www.teamrankings.com/nba/stat/average-4th-quarter-margin",
                "https://www.teamrankings.com/nba/stat/average-1st-half-margin",
                "https://www.teamrankings.com/nba/stat/average-2nd-half-margin",
                "https://www.teamrankings.com/nba/stat/average-overtime-margin",
                "https://www.teamrankings.com/nba/stat/average-margin-thru-3-quarters",
                "https://www.teamrankings.com/nba/stat/points-from-2-pointers",
                "https://www.teamrankings.com/nba/stat/points-from-3-pointers",
                "https://www.teamrankings.com/nba/stat/percent-of-points-from-2-pointers",
                "https://www.teamrankings.com/nba/stat/percent-of-points-from-3-pointers",
                "https://www.teamrankings.com/nba/stat/percent-of-points-from-free-throws",
                "https://www.teamrankings.com/nba/stat/shooting-pct",
                "https://www.teamrankings.com/nba/stat/effective-field-goal-pct",
                "https://www.teamrankings.com/nba/stat/three-point-pct",
                "https://www.teamrankings.com/nba/stat/two-point-pct",
                "https://www.teamrankings.com/nba/stat/free-throw-pct",
                "https://www.teamrankings.com/nba/stat/true-shooting-percentage",
                "https://www.teamrankings.com/nba/stat/field-goals-made-per-game",
                "https://www.teamrankings.com/nba/stat/field-goals-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/three-pointers-made-per-game",
                "https://www.teamrankings.com/nba/stat/three-pointers-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/free-throws-made-per-game",
                "https://www.teamrankings.com/nba/stat/free-throws-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/three-point-rate",
                "https://www.teamrankings.com/nba/stat/two-point-rate",
                "https://www.teamrankings.com/nba/stat/fta-per-fga",
                "https://www.teamrankings.com/nba/stat/ftm-per-100-possessions",
                "https://www.teamrankings.com/nba/stat/free-throw-rate",
                "https://www.teamrankings.com/nba/stat/non-blocked-2-pt-pct",
                "https://www.teamrankings.com/nba/stat/offensive-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/defensive-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/team-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/total-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/offensive-rebounding-pct",
                "https://www.teamrankings.com/nba/stat/defensive-rebounding-pct",
                "https://www.teamrankings.com/nba/stat/total-rebounding-percentage",
                "https://www.teamrankings.com/nba/stat/blocks-per-game",
                "https://www.teamrankings.com/nba/stat/steals-per-game",
                "https://www.teamrankings.com/nba/stat/block-pct",
                "https://www.teamrankings.com/nba/stat/steal-pct",
                "https://www.teamrankings.com/nba/stat/assists-per-game",
                "https://www.teamrankings.com/nba/stat/turnovers-per-game",
                "https://www.teamrankings.com/nba/stat/turnovers-per-possession",
                "https://www.teamrankings.com/nba/stat/assist--per--turnover-ratio",
                "https://www.teamrankings.com/nba/stat/assists-per-fgm",
                "https://www.teamrankings.com/nba/stat/assists-per-possession",
                "https://www.teamrankings.com/nba/stat/turnover-pct",
                "https://www.teamrankings.com/nba/stat/personal-fouls-per-game",
                "https://www.teamrankings.com/nba/stat/technical-fouls-per-game",
                "https://www.teamrankings.com/nba/stat/personal-fouls-per-possession",
                "https://www.teamrankings.com/nba/stat/personal-foul-pct",
                "https://www.teamrankings.com/nba/stat/opponent-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-average-scoring-margin",
                "https://www.teamrankings.com/nba/stat/defensive-efficiency",
                "https://www.teamrankings.com/nba/stat/opponent-floor-percentage",
                "https://www.teamrankings.com/nba/stat/opponent-1st-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-2nd-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-3rd-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-4th-quarter-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-overtime-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-points-in-paint-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-fastbreak-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-fastbreak-efficiency",
                "https://www.teamrankings.com/nba/stat/opponent-average-biggest-lead",
                "https://www.teamrankings.com/nba/stat/opponent-1st-half-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-2nd-half-points-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-points-from-2-pointers",
                "https://www.teamrankings.com/nba/stat/opponent-points-from-3-pointers",
                "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-2-pointers",
                "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-3-pointers",
                "https://www.teamrankings.com/nba/stat/opponent-percent-of-points-from-free-throws",
                "https://www.teamrankings.com/nba/stat/opponent-shooting-pct",
                "https://www.teamrankings.com/nba/stat/opponent-effective-field-goal-pct",
                "https://www.teamrankings.com/nba/stat/opponent-three-point-pct",
                "https://www.teamrankings.com/nba/stat/opponent-two-point-pct",
                "https://www.teamrankings.com/nba/stat/opponent-free-throw-pct",
                "https://www.teamrankings.com/nba/stat/opponent-true-shooting-percentage",
                "https://www.teamrankings.com/nba/stat/opponent-field-goals-made-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-field-goals-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-three-pointers-made-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-three-pointers-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-free-throws-made-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-free-throws-attempted-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-three-point-rate",
                "https://www.teamrankings.com/nba/stat/opponent-two-point-rate",
                "https://www.teamrankings.com/nba/stat/opponent-fta-per-fga",
                "https://www.teamrankings.com/nba/stat/opponent-ftm-per-100-possessions",
                "https://www.teamrankings.com/nba/stat/opponent-free-throw-rate",
                "https://www.teamrankings.com/nba/stat/opponent-non-blocked-2-pt-pct",
                "https://www.teamrankings.com/nba/stat/opponent-offensive-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-defensive-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-team-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-total-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-offensive-rebounding-pct",
                "https://www.teamrankings.com/nba/stat/opponent-defensive-rebounding-pct",
                "https://www.teamrankings.com/nba/stat/opponent-blocks-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-steals-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-block-pct",
                "https://www.teamrankings.com/nba/stat/opponent-steals-perpossession",
                "https://www.teamrankings.com/nba/stat/opponent-steal-pct",
                "https://www.teamrankings.com/nba/stat/opponent-assists-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-turnovers-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-assist--per--turnover-ratio",
                "https://www.teamrankings.com/nba/stat/opponent-assists-per-fgm",
                "https://www.teamrankings.com/nba/stat/opponent-assists-per-possession",
                "https://www.teamrankings.com/nba/stat/opponent-turnovers-per-possession",
                "https://www.teamrankings.com/nba/stat/opponent-turnover-pct",
                "https://www.teamrankings.com/nba/stat/opponent-personal-fouls-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-technical-fouls-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-personal-fouls-per-possession",
                "https://www.teamrankings.com/nba/stat/opponent-personal-foul-pct",
                "https://www.teamrankings.com/nba/stat/games-played",
                "https://www.teamrankings.com/nba/stat/possessions-per-game",
                "https://www.teamrankings.com/nba/stat/extra-chances-per-game",
                "https://www.teamrankings.com/nba/stat/effective-possession-ratio",
                "https://www.teamrankings.com/nba/stat/opponent-effective-possession-ratio",
                "https://www.teamrankings.com/nba/stat/points-plus-rebounds-plus-assists-per-game",
                "https://www.teamrankings.com/nba/stat/points-plus-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/points-plus-assists-per-game",
                "https://www.teamrankings.com/nba/stat/rebounds-plus-assists-per-game",
                "https://www.teamrankings.com/nba/stat/steals-plus-blocks-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-points-plus-rebounds-plus-assists-per-gam",
                "https://www.teamrankings.com/nba/stat/opponent-points-plus-rebounds-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-points-plus-assists-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-rebounds-plus-assists-per-game",
                "https://www.teamrankings.com/nba/stat/opponent-steals-plus-blocks-per-game",
                "https://www.teamrankings.com/nba/stat/win-pct-all-games",
                "https://www.teamrankings.com/nba/stat/win-pct-close-games",
                "https://www.teamrankings.com/nba/stat/opponent-win-pct-all-games",
                "https://www.teamrankings.com/nba/stat/opponent-win-pct-close-games",
                "https://www.teamrankings.com/nba/ranking/predictive-by-other",
                "https://www.teamrankings.com/nba/ranking/home-by-other",
                "https://www.teamrankings.com/nba/ranking/away-by-other",
                "https://www.teamrankings.com/nba/ranking/home-adv-by-other",
                "https://www.teamrankings.com/nba/ranking/schedule-strength-by-other",
                "https://www.teamrankings.com/nba/ranking/future-sos-by-other",
                "https://www.teamrankings.com/nba/ranking/season-sos-by-other",
                "https://www.teamrankings.com/nba/ranking/sos-basic-by-other",
                "https://www.teamrankings.com/nba/ranking/in-division-sos-by-other",
                "https://www.teamrankings.com/nba/ranking/non-division-sos-by-other",
                "https://www.teamrankings.com/nba/ranking/last-5-games-by-other",
                "https://www.teamrankings.com/nba/ranking/last-10-games-by-other",
                "https://www.teamrankings.com/nba/ranking/in-division-by-other",
                "https://www.teamrankings.com/nba/ranking/non-division-by-other",
                "https://www.teamrankings.com/nba/ranking/luck-by-other",
                "https://www.teamrankings.com/nba/ranking/consistency-by-other",
                "https://www.teamrankings.com/nba/ranking/vs-1-5-by-other",
                "https://www.teamrankings.com/nba/ranking/vs-6-10-by-other",
                "https://www.teamrankings.com/nba/ranking/vs-11-16-by-other",
                "https://www.teamrankings.com/nba/ranking/vs-17-22-by-other",
                "https://www.teamrankings.com/nba/ranking/vs-23-30-by-other",
                "https://www.teamrankings.com/nba/ranking/first-half-by-other",
                "https://www.teamrankings.com/nba/ranking/second-half-by-other"
                ],
            'NFL': [
                "https://www.teamrankings.com/nfl/stat/points-per-game",
                "https://www.teamrankings.com/nfl/stat/average-scoring-margin",
                "https://www.teamrankings.com/nfl/stat/yards-per-point",
                "https://www.teamrankings.com/nfl/stat/yards-per-point-margin",
                "https://www.teamrankings.com/nfl/stat/points-per-play",
                "https://www.teamrankings.com/nfl/stat/points-per-play-margin",
                "https://www.teamrankings.com/nfl/stat/touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/red-zone-scoring-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/red-zone-scores-per-game",
                "https://www.teamrankings.com/nfl/stat/red-zone-scoring-pct",
                "https://www.teamrankings.com/nfl/stat/extra-point-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/extra-points-made-per-game",
                "https://www.teamrankings.com/nfl/stat/two-point-conversion-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/two-point-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/points-per-field-goal-attempt",
                "https://www.teamrankings.com/nfl/stat/extra-point-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/two-point-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/offensive-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/defensive-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/special-teams-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/offensive-points-per-game",
                "https://www.teamrankings.com/nfl/stat/defensive-points-per-game",
                "https://www.teamrankings.com/nfl/stat/special-teams-points-per-game",
                "https://www.teamrankings.com/nfl/stat/offensive-point-share-pct",
                "https://www.teamrankings.com/nfl/stat/1st-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/2nd-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/3rd-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/4th-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/overtime-points-per-game",
                "https://www.teamrankings.com/nfl/stat/1st-half-points-per-game",
                "https://www.teamrankings.com/nfl/stat/2nd-half-points-per-game",
                "https://www.teamrankings.com/nfl/stat/1st-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/2nd-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/3rd-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/4th-quarter-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/1st-half-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/2nd-half-time-of-possession-share-pct",
                "https://www.teamrankings.com/nfl/stat/yards-per-game",
                "https://www.teamrankings.com/nfl/stat/plays-per-game",
                "https://www.teamrankings.com/nfl/stat/yards-per-play",
                "https://www.teamrankings.com/nfl/stat/first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/third-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/third-down-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/fourth-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/fourth-down-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/average-time-of-possession-net-of-ot",
                "https://www.teamrankings.com/nfl/stat/time-of-possession-pct-net-of-ot",
                "https://www.teamrankings.com/nfl/stat/seconds-per-play",
                "https://www.teamrankings.com/nfl/stat/first-downs-per-play",
                "https://www.teamrankings.com/nfl/stat/third-down-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/fourth-down-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/punts-per-play",
                "https://www.teamrankings.com/nfl/stat/punts-per-offensive-score",
                "https://www.teamrankings.com/nfl/stat/opponent-tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-solo-tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-assisted-tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/rushing-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/rushing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/rushing-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/rushing-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/yards-per-rush-attempt",
                "https://www.teamrankings.com/nfl/stat/rushing-play-pct",
                "https://www.teamrankings.com/nfl/stat/rushing-touchdown-pct",
                "https://www.teamrankings.com/nfl/stat/rushing-first-down-pct",
                "https://www.teamrankings.com/nfl/stat/rushing-yards-pct",
                "https://www.teamrankings.com/nfl/stat/pass-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/completions-per-game",
                "https://www.teamrankings.com/nfl/stat/incompletions-per-game",
                "https://www.teamrankings.com/nfl/stat/completion-pct",
                "https://www.teamrankings.com/nfl/stat/passing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/gross-passing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/yards-per-pass-attempt",
                "https://www.teamrankings.com/nfl/stat/yards-per-completion",
                "https://www.teamrankings.com/nfl/stat/passing-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/passing-touchdown-pct",
                "https://www.teamrankings.com/nfl/stat/qb-sacked-per-game",
                "https://www.teamrankings.com/nfl/stat/qb-sacked-pct",
                "https://www.teamrankings.com/nfl/stat/passing-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/passing-first-down-pct",
                "https://www.teamrankings.com/nfl/stat/average-team-passer-rating",
                "https://www.teamrankings.com/nfl/stat/passing-play-pct",
                "https://www.teamrankings.com/nfl/stat/passing-yards-pct",
                "https://www.teamrankings.com/nfl/stat/other-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/field-goal-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/field-goals-made-per-game",
                "https://www.teamrankings.com/nfl/stat/field-goals-got-blocked-per-game",
                "https://www.teamrankings.com/nfl/stat/kicking-points-per-game",
                "https://www.teamrankings.com/nfl/stat/punt-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/punts-got-blocked-per-game",
                "https://www.teamrankings.com/nfl/stat/gross-punt-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/net-punt-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/kickoffs-per-game",
                "https://www.teamrankings.com/nfl/stat/touchbacks-per-game",
                "https://www.teamrankings.com/nfl/stat/kickoff-touchback-pct",
                "https://www.teamrankings.com/nfl/stat/field-goal-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/field-goal-got-blocked-pct",
                "https://www.teamrankings.com/nfl/stat/field-goal-conversion-pct-net-of-blocks",
                "https://www.teamrankings.com/nfl/stat/punt-blocked-pct",
                "https://www.teamrankings.com/nfl/stat/net-yards-per-punt-attempt",
                "https://www.teamrankings.com/nfl/stat/gross-yards-per-successful-punt",
                "https://www.teamrankings.com/nfl/stat/net-yards-per-successful-punt",
                "https://www.teamrankings.com/nfl/stat/opponent-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opp-yards-per-point",
                "https://www.teamrankings.com/nfl/stat/opponent-points-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-red-zone-scores-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-red-zone-scoring-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-extra-point-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-extra-points-made-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-two-point-conversion-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-two-point-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-points-per-field-goal-attempt",
                "https://www.teamrankings.com/nfl/stat/opponent-extra-point-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-two-point-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-offensive-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-defensive-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-special-teams-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-offensive-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-defensive-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-special-teams-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-offensive-point-share-pct",
                "https://www.teamrankings.com/nfl/stat/opp-1st-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opp-2nd-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opp-3rd-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opp-4th-quarter-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opp-overtime-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-1st-half-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-2nd-half-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-plays-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-yards-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-third-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-third-down-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-fourth-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-fourth-down-conversions-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-average-time-of-possession-net-of-ot",
                "https://www.teamrankings.com/nfl/stat/opponent-time-of-possession-pct-net-of-ot",
                "https://www.teamrankings.com/nfl/stat/opponent-seconds-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-first-downs-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-third-down-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-fourth-down-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-punts-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-punts-per-offensive-score",
                "https://www.teamrankings.com/nfl/stat/tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/solo-tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/assisted-tackles-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-yards-per-rush-attempt",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-play-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-touchdown-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-first-down-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-rushing-yards-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-pass-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-completions-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-incompletions-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-completion-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-gross-passing-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-yards-per-pass-attempt",
                "https://www.teamrankings.com/nfl/stat/opponent-yards-per-completion",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-touchdown-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-average-team-passer-rating",
                "https://www.teamrankings.com/nfl/stat/sack-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-play-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-yards-pct",
                "https://www.teamrankings.com/nfl/stat/sacks-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-passing-first-down-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-other-touchdowns-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-field-goal-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-field-goals-made-per-game",
                "https://www.teamrankings.com/nfl/stat/field-goals-blocked-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-kicking-points-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-punt-attempts-per-game",
                "https://www.teamrankings.com/nfl/stat/punts-blocked-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-gross-punt-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-net-punt-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-kickoffs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-touchbacks-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-kickoff-touchback-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-field-goal-conversion-pct",
                "https://www.teamrankings.com/nfl/stat/block-field-goal-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-field-goal-conversion-pct-net-of-blocks",
                "https://www.teamrankings.com/nfl/stat/block-punt-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-net-yards-per-punt-attempt",
                "https://www.teamrankings.com/nfl/stat/opponent-gross-yards-per-successful-punt",
                "https://www.teamrankings.com/nfl/stat/opponent-net-yards-per-successful-punt",
                "https://www.teamrankings.com/nfl/stat/interceptions-thrown-per-game",
                "https://www.teamrankings.com/nfl/stat/percent-of-games-with-an-interception-thrown",
                "https://www.teamrankings.com/nfl/stat/fumbles-per-game",
                "https://www.teamrankings.com/nfl/stat/fumbles-lost-per-game",
                "https://www.teamrankings.com/nfl/stat/fumbles-not-lost-per-game",
                "https://www.teamrankings.com/nfl/stat/safeties-per-game",
                "https://www.teamrankings.com/nfl/stat/giveaways-per-game",
                "https://www.teamrankings.com/nfl/stat/turnover-margin-per-game",
                "https://www.teamrankings.com/nfl/stat/interceptions-per-game",
                "https://www.teamrankings.com/nfl/stat/percent-of-games-with-an-interception",
                "https://www.teamrankings.com/nfl/stat/opponent-fumbles-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-fumbles-lost-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-fumbles-not-lost-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-safeties-per-game",
                "https://www.teamrankings.com/nfl/stat/takeaways-per-game",
                "https://www.teamrankings.com/nfl/stat/pass-intercepted-pct",
                "https://www.teamrankings.com/nfl/stat/fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/giveaway-fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/takeaway-fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/interception-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-giveaway-fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/opponent-takeaway-fumble-recovery-pct",
                "https://www.teamrankings.com/nfl/stat/penalties-per-game",
                "https://www.teamrankings.com/nfl/stat/penalty-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/penalty-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-penalties-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-penalty-yards-per-game",
                "https://www.teamrankings.com/nfl/stat/opponent-penalty-first-downs-per-game",
                "https://www.teamrankings.com/nfl/stat/penalty-yards-per-penalty",
                "https://www.teamrankings.com/nfl/stat/penalties-per-play",
                "https://www.teamrankings.com/nfl/stat/opponent-penalty-yards-per-penalty",
                "https://www.teamrankings.com/nfl/stat/opponent-penalties-per-play",
                "https://www.teamrankings.com/nfl/ranking/predictive-by-other",
                "https://www.teamrankings.com/nfl/ranking/home-by-other",
                "https://www.teamrankings.com/nfl/ranking/away-by-other",
                "https://www.teamrankings.com/nfl/ranking/home-adv-by-other",
                "https://www.teamrankings.com/nfl/ranking/schedule-strength-by-other",
                "https://www.teamrankings.com/nfl/ranking/future-sos-by-other",
                "https://www.teamrankings.com/nfl/ranking/season-sos-by-other",
                "https://www.teamrankings.com/nfl/ranking/sos-basic-by-other",
                "https://www.teamrankings.com/nfl/ranking/in-division-sos-by-other",
                "https://www.teamrankings.com/nfl/ranking/non-division-sos-by-other",
                "https://www.teamrankings.com/nfl/ranking/last-5-games-by-other",
                "https://www.teamrankings.com/nfl/ranking/last-10-games-by-other",
                "https://www.teamrankings.com/nfl/ranking/in-division-by-other",
                "https://www.teamrankings.com/nfl/ranking/non-division-by-other",
                "https://www.teamrankings.com/nfl/ranking/luck-by-other",
                "https://www.teamrankings.com/nfl/ranking/consistency-by-other",
                "https://www.teamrankings.com/nfl/ranking/vs-1-5-by-other",
                "https://www.teamrankings.com/nfl/ranking/vs-6-10-by-other",
                "https://www.teamrankings.com/nfl/ranking/vs-11-16-by-other",
                "https://www.teamrankings.com/nfl/ranking/vs-17-22-by-other",
                "https://www.teamrankings.com/nfl/ranking/vs-23-32-by-other",
                "https://www.teamrankings.com/nfl/ranking/first-half-by-other",
                "https://www.teamrankings.com/nfl/ranking/second-half-by-other"
                ],
            'MLB': [
                "https://www.teamrankings.com/mlb/stat/runs-per-game",
                "https://www.teamrankings.com/mlb/stat/at-bats-per-game",
                "https://www.teamrankings.com/mlb/stat/hits-per-game",
                "https://www.teamrankings.com/mlb/stat/home-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/singles-per-game",
                "https://www.teamrankings.com/mlb/stat/doubles-per-game",
                "https://www.teamrankings.com/mlb/stat/triples-per-game",
                "https://www.teamrankings.com/mlb/stat/rbis-per-game",
                "https://www.teamrankings.com/mlb/stat/walks-per-game",
                "https://www.teamrankings.com/mlb/stat/strikeouts-per-game",
                "https://www.teamrankings.com/mlb/stat/stolen-bases-per-game",
                "https://www.teamrankings.com/mlb/stat/stolen-bases-attempted-per-game",
                "https://www.teamrankings.com/mlb/stat/caught-stealing-per-game",
                "https://www.teamrankings.com/mlb/stat/sacrifice-hits-per-game",
                "https://www.teamrankings.com/mlb/stat/sacrifice-flys-per-game",
                "https://www.teamrankings.com/mlb/stat/left-on-base-per-game",
                "https://www.teamrankings.com/mlb/stat/team-left-on-base-per-game",
                "https://www.teamrankings.com/mlb/stat/hit-by-pitch-per-game",
                "https://www.teamrankings.com/mlb/stat/grounded-into-double-plays-per-game",
                "https://www.teamrankings.com/mlb/stat/runners-left-in-scoring-position-per-game",
                "https://www.teamrankings.com/mlb/stat/total-bases-per-game",
                "https://www.teamrankings.com/mlb/stat/batting-average",
                "https://www.teamrankings.com/mlb/stat/slugging-pct",
                "https://www.teamrankings.com/mlb/stat/on-base-pct",
                "https://www.teamrankings.com/mlb/stat/on-base-plus-slugging-pct",
                "https://www.teamrankings.com/mlb/stat/plate-appearances",
                "https://www.teamrankings.com/mlb/stat/run-differential",
                "https://www.teamrankings.com/mlb/stat/batting-average-on-balls-in-play",
                "https://www.teamrankings.com/mlb/stat/isolated-power",
                "https://www.teamrankings.com/mlb/stat/secondary-average",
                "https://www.teamrankings.com/mlb/stat/at-bats-per-home-run",
                "https://www.teamrankings.com/mlb/stat/home-run-pct",
                "https://www.teamrankings.com/mlb/stat/strikeout-pct",
                "https://www.teamrankings.com/mlb/stat/walk-pct",
                "https://www.teamrankings.com/mlb/stat/extra-base-hit-pct",
                "https://www.teamrankings.com/mlb/stat/hits-for-extra-bases-pct",
                "https://www.teamrankings.com/mlb/stat/stolen-base-pct",
                "https://www.teamrankings.com/mlb/stat/hits-per-run",
                "https://www.teamrankings.com/mlb/stat/outs-pitched-per-game",
                "https://www.teamrankings.com/mlb/stat/earned-runs-against-per-game",
                "https://www.teamrankings.com/mlb/stat/earned-run-average",
                "https://www.teamrankings.com/mlb/stat/walks-plus-hits-per-inning-pitched",
                "https://www.teamrankings.com/mlb/stat/strikeouts-per-9",
                "https://www.teamrankings.com/mlb/stat/hits-per-9",
                "https://www.teamrankings.com/mlb/stat/home-runs-per-9",
                "https://www.teamrankings.com/mlb/stat/walks-per-9",
                "https://www.teamrankings.com/mlb/stat/strikeouts-per-walk",
                "https://www.teamrankings.com/mlb/stat/shutouts",
                "https://www.teamrankings.com/mlb/stat/double-plays-per-game",
                "https://www.teamrankings.com/mlb/stat/errors-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-at-bats-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-hits-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-home-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-singles-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-doubles-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-triples-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-rbis-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-walks-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-strikeouts-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-stolen-bases-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-stolen-bases-attempted",
                "https://www.teamrankings.com/mlb/stat/opponent-caught-stealing-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-sacrifice-hits-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-sacrifice-flys-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-left-on-base-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-team-left-on-base-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-hit-by-pitch-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-grounded-into-double-plays-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-runners-left-in-scoring-position-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-total-bases-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-batting-average",
                "https://www.teamrankings.com/mlb/stat/opponent-slugging-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-on-base-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-on-base-plus-slugging-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-plate-appearances",
                "https://www.teamrankings.com/mlb/stat/opponent-run-differential",
                "https://www.teamrankings.com/mlb/stat/opponent-batting-average-on-balls-in-play",
                "https://www.teamrankings.com/mlb/stat/opponent-isolated-power",
                "https://www.teamrankings.com/mlb/stat/opponent-secondary-average",
                "https://www.teamrankings.com/mlb/stat/opponent-at-bats-per-home-run",
                "https://www.teamrankings.com/mlb/stat/opponent-home-run-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-strikeout-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-walk-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-extra-base-hit-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-hits-for-extra-bases-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-stolen-base-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-hits-per-run",
                "https://www.teamrankings.com/mlb/stat/opponent-outs-pitched-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-earned-runs-against-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-earned-run-average",
                "https://www.teamrankings.com/mlb/stat/opponent-walks-plus-hits--per--innings-pitched",
                "https://www.teamrankings.com/mlb/stat/opponent-strikeouts-per-9",
                "https://www.teamrankings.com/mlb/stat/opponent-hits-per-9",
                "https://www.teamrankings.com/mlb/stat/opponent-home-runs-per-9",
                "https://www.teamrankings.com/mlb/stat/opponent-walks-per-9",
                "https://www.teamrankings.com/mlb/stat/opponent-strikeouts-per-walk",
                "https://www.teamrankings.com/mlb/stat/opponent-shutouts",
                "https://www.teamrankings.com/mlb/stat/opponent-double-plays-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-errors-per-game",
                "https://www.teamrankings.com/mlb/stat/games-played",
                "https://www.teamrankings.com/mlb/stat/yes-run-first-inning-pct",
                "https://www.teamrankings.com/mlb/stat/no-run-first-inning-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-yes-run-first-inning-pct",
                "https://www.teamrankings.com/mlb/stat/opponent-no-run-first-inning-pct",
                "https://www.teamrankings.com/mlb/stat/1st-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/2nd-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/3rd-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/4th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/5th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/6th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/7th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/8th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/9th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/extra-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-1st-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-2nd-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-3rd-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-4th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-5th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-6th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-7th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-8th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-9th-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-extra-inning-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/first-4-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/first-5-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/first-6-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/last-2-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/last-3-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/last-4-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-first-4-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-first-5-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-first-6-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-last-2-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-last-3-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/opponent-last-4-innings-runs-per-game",
                "https://www.teamrankings.com/mlb/stat/win-pct-all-games",
                "https://www.teamrankings.com/mlb/stat/win-pct-close-games",
                "https://www.teamrankings.com/mlb/stat/opponent-win-pct-all-games",
                "https://www.teamrankings.com/mlb/stat/opponent-win-pct-close-games",
                "https://www.teamrankings.com/mlb/ranking/predictive-by-other",
                "https://www.teamrankings.com/mlb/ranking/home-by-other",
                "https://www.teamrankings.com/mlb/ranking/away-by-other",
                "https://www.teamrankings.com/mlb/ranking/home-adv-by-other",
                "https://www.teamrankings.com/mlb/ranking/schedule-strength-by-other",
                "https://www.teamrankings.com/mlb/ranking/future-sos-by-other",
                "https://www.teamrankings.com/mlb/ranking/season-sos-by-other",
                "https://www.teamrankings.com/mlb/ranking/sos-basic-by-other",
                "https://www.teamrankings.com/mlb/ranking/in-division-sos-by-other",
                "https://www.teamrankings.com/mlb/ranking/non-division-sos-by-other",
                "https://www.teamrankings.com/mlb/ranking/last-5-games-by-other",
                "https://www.teamrankings.com/mlb/ranking/last-10-games-by-other",
                "https://www.teamrankings.com/mlb/ranking/in-division-by-other",
                "https://www.teamrankings.com/mlb/ranking/non-division-by-other",
                "https://www.teamrankings.com/mlb/ranking/luck-by-other",
                "https://www.teamrankings.com/mlb/ranking/consistency-by-other",
                "https://www.teamrankings.com/mlb/ranking/vs-1-5-by-other",
                "https://www.teamrankings.com/mlb/ranking/vs-6-10-by-other",
                "https://www.teamrankings.com/mlb/ranking/vs-11-16-by-other",
                "https://www.teamrankings.com/mlb/ranking/vs-17-22-by-other",
                "https://www.teamrankings.com/mlb/ranking/vs-23-30-by-other"
                ]
        }
        if self.sport not in url_lists:
            raise ValueError(f"No stat pages for {self.sport}")

        # first time: scrape all tables into cache
        if not self._stats_cache:
            self._stats_cache = {}
            for url in url_lists[self.sport]:
                dated_url = f"{url}?date={self.date.isoformat()}"
                while True:
                    resp = requests.get(dated_url, headers={"User-Agent":"Mozilla/5.0"})
                    try:
                        resp.raise_for_status()
                        break
                    except HTTPError as e:
                        # if blocked, wait 4 minutes and retry
                        if resp.status_code == 403:
                            logging.warning(f"403 Forbidden at {dated_url}, sleeping 4 minutes before retry")
                            time.sleep(4 * 60)
                            continue
                        elif resp.status_code == 504:
                            logging.warning(f"504 Gateway Timeout at {dated_url}, sleeping 4 minutes before retry")
                            time.sleep(4 * 60)
                            continue
                        # other HTTP errors bubble up
                        raise
                soup = BeautifulSoup(resp.text, "html.parser")
                table = soup.find("table")
                if not table:
                    raise RuntimeError(f"No table found at {dated_url}")

                # get column headers
                headers = [th.get_text(strip=True) for th in table.select("thead th")]
                # parse rows
                raw_rows = {}
                numeric_rows = []
                for tr in table.select("tbody tr"):
                    tds = tr.select("td")
                    row = {}
                    for i, h in enumerate(headers):
                        text = tds[i].get_text(strip=True) if h != "Team" else (
                            tds[i].find("a").get_text(strip=True) if tds[i].find("a") else tds[i].get_text(strip=True)
                        )
                        row[h] = text
                    # numeric parse for normalization
                    nums = {h: self._parse_stat_value(row[h]) for h in headers if h != "Team"}
                    numeric_rows.append(nums)
                    raw_rows[row["Team"]] = row

                if not raw_rows:
                    logging.error(f"No stats available on page {dated_url!r}, thus the day is ruined")
                    self._stats_cache_valid = False
                    return None

                # compute min/max per numeric column
                min_max = {
                    h: (min(r[h] for r in numeric_rows), max(r[h] for r in numeric_rows))
                    for h in numeric_rows[0]
                }

                # build per‐team raw & normalized dicts
                slug = url.rstrip("/").split("/")[-1]
                self._stats_cache[slug] = {}
                for team, raw in raw_rows.items():
                    raw_stats = {h: self._parse_stat_value(raw[h]) for h in headers if h != "Team"}
                    norm_stats = {}
                    for h, val in {h: self._parse_stat_value(raw[h]) for h in headers if h != "Team"}.items():
                        lo, hi = min_max[h]
                        norm_stats[h] = (val - lo) / (hi - lo) if hi > lo else 0.0
                    self._stats_cache[slug][team] = {
                        "raw": raw_stats,
                        "normalized": norm_stats
                    }

        # now build result for this team
        result = {}
        for slug, teams in self._stats_cache.items():
            if team_name not in teams:
                logging.error(f"Team {team_name!r} not found in stats for '{slug}'")
                return None
            result[slug] = teams[team_name]

        return result

    def get_player_stats(self, team_id: int):
        """
        Query database or scrape stats for all players on the given team for the specified date.
        Returns:
            List[Dict]: [{"player_id": id, "stats": {stat_name: stat_value, ...}}, ...]
        """
        pass

    def get_betting_odds(self, game_id: str) -> dict:
        """
        Fetches closing odds for moneyline, spread, and total (over/under)
        for the given ESPN event ID. Returns a dict:
            {
              'team1_moneyline':     int or None,
              'team2_moneyline':     int or None,
              'team1_spread':        float or None,
              'team2_spread':        float or None,
              'team1_spread_odds':   int or None,
              'team2_spread_odds':   int or None,
              'total_score':         float or None,
              'over_odds':           int or None,
              'under_odds':          int or None
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
                'team1_moneyline':    None,
                'team2_moneyline':    None,
                'team1_spread':       None,
                'team2_spread':       None,
                'team1_spread_odds':  None,
                'team2_spread_odds':  None,
                'total_score':        None,
                'over_odds':          None,
                'under_odds':         None
            }

        # pick the provider with the highest priority
        entry = max(items, key=lambda e: e.get('provider', {}).get('priority', -999))

        away = entry.get('awayTeamOdds', {})  # team1 = away
        home = entry.get('homeTeamOdds', {})  # team2 = home

        # 1) Moneylines (straight moneyline odds)
        team1_ml = away.get('moneyLine')
        team2_ml = home.get('moneyLine')

        # 2) Spread (point spread)
        raw_spread = entry.get('spread')
        if isinstance(raw_spread, (int, float)):
            # raw_spread is the home-team spread (positive means home favored by that many)
            team2_sp = raw_spread
            team1_sp = -raw_spread
        else:
            # fallback to spreadOdds if no top-level spread
            team1_sp = away.get('spreadOdds')
            team2_sp = home.get('spreadOdds')

        # 3) Closing spread odds (american)
        def _extract_close_spread_odds(team_odds: dict) -> int | None:
            """
            Look under team_odds['close']['spread']['american'] if available,
            else return None.
            """
            try:
                close_info = team_odds.get('close', {})
                spread_info = close_info.get('spread', {})
                american = spread_info.get('american')
                if american is None:
                    return None
                # american might be a string like "+160" or "-105"
                return int(str(american).replace("+", "").replace("−", "-"))
            except Exception:
                return None

        team1_sp_odds = _extract_close_spread_odds(away)
        team2_sp_odds = _extract_close_spread_odds(home)

        # 4) Total (over/under)
        total = entry.get('overUnder')

        # 5) Closing over/under odds (american)
        def _extract_close_total_odds(side: str) -> int | None:
            """
            side should be 'over' or 'under'. We look under entry['close'][side]['american'].
            """
            try:
                close_block = entry.get('close', {})
                side_info = close_block.get(side, {})
                american = side_info.get('american')
                if american is None:
                    return None
                return int(str(american).replace("+", "").replace("−", "-"))
            except Exception:
                return None

        over_odds = _extract_close_total_odds('over')
        under_odds = _extract_close_total_odds('under')

        return {
            'team1_moneyline':    team1_ml,
            'team2_moneyline':    team2_ml,
            'team1_spread':       team1_sp,
            'team2_spread':       team2_sp,
            'team1_spread_odds':  team1_sp_odds,
            'team2_spread_odds':  team2_sp_odds,
            'total_score':        total,
            'over_odds':          over_odds,
            'under_odds':         under_odds
        }

    def get_weather_info(self, time_iso: str, city: str, state: str = None, country: str = None) -> dict:
        """
        Fetch weather data for the given city/state/country at the date+time 
        of this Pregame instance. Works for past (archive) or future (forecast).

        time_iso: e.g. "18:00:00Z"
        city/state/country: for geocoding

        Returns a dict mapping:
            {
              "temperature": float,            # °C
              "relative_humidity": float,      # %
              "dewpoint": float,               # °C
              "apparent_temperature": float,   # °C
              "precipitation": float,          # mm
              "rain": float,                   # mm
              "snowfall": float,               # cm
              "snow_depth": float,             # cm
              "sea_level_pressure": float,     # hPa
              "surface_pressure": float,       # hPa
              "cloud_cover": float,            # %
              "evapotranspiration": float,     # mm
              "vapour_pressure_deficit": float,# hPa
              "wind_speed": float,             # km/h
              "wind_direction": float,         # °
              "wind_gusts": float,             # km/h
            }
        """
        # 1) Geocode
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": city, "language": "en", "limit": 1}
        if state:   params["admin1"] = state
        if country: params["country"] = country

        try:
            r = requests.get(geocode_url, params=params, timeout=10)
            r.raise_for_status()
            results = r.json().get("results") or []
            if not results and state:
                logging.warning(f"Geocoding failed for '{city}, {state}, {country}', retrying with state only")
                # retry using only state
                params_retry = {"name": state, "language": "en", "limit": 1}
                if country: params_retry["country"] = country
                r2 = requests.get(geocode_url, params=params_retry, timeout=10)
                r2.raise_for_status()
                results = r2.json().get("results") or []
                if results:
                    logging.info(f"Geocoding succeeded on retry for '{state}, {country}'")
                else:
                    logging.error(f"Geocoding still failed for '{state}, {country}'")
            if not results:
                logging.error(f"Geocoding ultimately failed for {city}, {state}, {country}")
                return {}
            loc = results[0]
            lat, lon = loc["latitude"], loc["longitude"]
        except Exception as e:
            logging.error(f"Error during geocoding: {e}")
            return {}

        # 2) Build UTC datetime for the requested hour
        time_part = time_iso.rstrip("Z")
        dt = datetime.fromisoformat(f"{self.date.isoformat()}T{time_part}")
        dt = dt.replace(tzinfo=timezone.utc)
        now_utc = datetime.now(timezone.utc)

        # 3) Choose endpoint (subtract 24 hours because archive doesnt update immediately, so just use forecast if the game happened within the last 24 hours)
        if dt < now_utc - timedelta(hours=24):
            weather_url = "https://archive-api.open-meteo.com/v1/archive"
        else:
            weather_url = "https://api.open-meteo.com/v1/forecast"

        # 4) Prepare all requested hourly variables
        vars_map = {
            "temperature":                 "temperature_2m",
            "relative_humidity":           "relativehumidity_2m",
            "dewpoint":                    "dewpoint_2m",
            "apparent_temperature":        "apparent_temperature",
            "precipitation":               "precipitation",
            "rain":                        "rain",
            "snowfall":                    "snowfall",
            "snow_depth":                  "snow_depth",
            "sea_level_pressure":          "pressure_msl",
            "surface_pressure":            "surface_pressure",
            "cloud_cover":                 "cloudcover",
            "evapotranspiration":          "et0_fao_evapotranspiration",
            "vapour_pressure_deficit":     "vapour_pressure_deficit",
            "wind_speed":                  "windspeed_10m",
            "wind_direction":              "winddirection_10m",
            "wind_gusts":                  "windgusts_10m",
        }

        date_str = dt.date().isoformat()
        weather_params = {
            "latitude":   lat,
            "longitude":  lon,
            "hourly":     ",".join(vars_map.values()),
            "start_date": date_str,
            "end_date":   date_str,
            "timezone":   "UTC",
        }

        # 5) Fetch
        try:
            r = requests.get(weather_url, params=weather_params, timeout=10)
            r.raise_for_status()
            hourly = r.json().get("hourly", {})
        except Exception as e:
            logging.error(f"Error fetching weather data: {e}")
            return {}

        times = hourly.get("time", [])
        dt = dt.replace(minute=0, second=0, microsecond=0)
        target = dt.strftime("%Y-%m-%dT%H:%M")
        if target not in times:
            logging.error(f"Hour {target} not found in weather data for {city}")
            return {}

        idx = times.index(target)
        output = {}
        for key, var in vars_map.items():
            if var in hourly:
                output[key] = hourly[var][idx]
            else:
                logging.warning(f"Variable {var!r} missing from response")
                output[key] = None

        return output

    def _fetch_scores_from_scoreboard(self) -> dict:
        """
        MODIFIED: This helper now uses the reliable scoreboard API to fetch all
        scores for the given date at once.

        Returns:
            A dictionary mapping game_id to its score, e.g.:
            {'4012345': {'team1_score': 10, 'team2_score': 3}, ...}
        """
        scores_map = {}
        try:
            # 1) Build ESPN scoreboard API URL (from your original working code)
            category, league = self._ESPN_MAP[self.sport]
            url = (
                f"https://site.api.espn.com/apis/site/v2/"
                f"sports/{category}/{league}/scoreboard"
            )
            date_str = self.date.strftime("%Y%m%d")

            # 2) Fetch scoreboard data
            resp = requests.get(url, params={"dates": date_str}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # 3) Loop through events and extract scores
            for event in data.get("events", []):
                game_id = event.get("id")
                if not game_id:
                    continue

                comp = event["competitions"][0]
                status = comp.get("status", {}).get("type", {})
                
                # Only process if game is completed
                if not status.get("completed", False):
                    continue

                # Map away=team1, home=team2
                teams = {c["homeAway"]: c for c in comp["competitors"]}
                away = teams.get("away")
                home = teams.get("home")

                if not away or not home:
                    continue
                
                try:
                    team1_score = int(away.get("score", 0))
                    team2_score = int(home.get("score", 0))
                    scores_map[game_id] = {
                        'team1_score': team1_score,
                        'team2_score': team2_score
                    }
                except (TypeError, ValueError):
                    logging.error(f"Invalid score format for game {game_id}")
                    continue
            
            return scores_map

        except Exception as e:
            logging.error(f"Failed to fetch or parse scoreboard for {self.sport} on {self.date}: {e}")
            return {} # Return empty dict on failure

    def update_final_scores_and_closing_odds(self) -> int:
        """
        Finds all games for the given date, fetches their final scores from the
        scoreboard, re-fetches the closing line odds, and updates the database.
        """
        # 1. Fetch all scores for the day in a single API call
        # This part is unchanged
        daily_scores = self._fetch_scores_from_scoreboard()

        if not daily_scores:
            logging.warning("Scoreboard was empty or could not be fetched. No scores to update.")
        
        cursor = self.conn.cursor()
        query = "SELECT game_id FROM games WHERE date = ? AND sport = ?"
        game_ids = [row[0] for row in cursor.execute(query, (self.date.isoformat(), self.sport))]

        if not game_ids:
            logging.info(f"No games found in DB for {self.sport} on {self.date} to post-process.")
            return 0

        # --- MODIFICATION START ---
        # Initialize counters for both skipped and successful updates
        skipped_count = 0
        updated_count = 0
        # --- MODIFICATION END ---

        for game_id in game_ids:
            # The logging and sleep are unchanged
            time.sleep(1) 

            final_scores = daily_scores.get(str(game_id))
            closing_odds = self.get_betting_odds(game_id)

            if not final_scores:
                skipped_count += 1
                continue
            
            # The update_data dictionary is unchanged
            update_data = {
                'team1_score': final_scores['team1_score'], 'team2_score': final_scores['team2_score'],
                'team1_moneyline': closing_odds.get('team1_moneyline'), 'team2_moneyline': closing_odds.get('team2_moneyline'),
                'team1_spread': closing_odds.get('team1_spread'), 'team2_spread': closing_odds.get('team2_spread'),
                'team1_spread_odds': closing_odds.get('team1_spread_odds'), 'team2_spread_odds': closing_odds.get('team2_spread_odds'),
                'total_score': closing_odds.get('total_score'), 'over_odds': closing_odds.get('over_odds'),
                'under_odds': closing_odds.get('under_odds'), 'game_id': game_id
            }

            # The SQL UPDATE statement is unchanged
            sql_update = """
                UPDATE games SET
                    team1_score = :team1_score, team2_score = :team2_score,
                    team1_moneyline = :team1_moneyline, team2_moneyline = :team2_moneyline,
                    team1_spread = :team1_spread, team2_spread = :team2_spread,
                    team1_spread_odds = :team1_spread_odds, team2_spread_odds = :team2_spread_odds,
                    total_score = :total_score, over_odds = :over_odds, under_odds = :under_odds
                WHERE game_id = :game_id
            """
            try:
                cursor.execute(sql_update, update_data)
                self.conn.commit()
                # --- MODIFICATION START ---
                # Increment the success counter after a successful commit
                updated_count += 1
                # --- MODIFICATION END ---
            except Exception as e:
                logging.error(f"Database update failed for game {game_id}: {e}")
                self.conn.rollback()
                skipped_count += 1
        
        # --- MODIFICATION START ---
        # Add the final summary log message before returning
        logging.info(f"Updated scores and odds for {updated_count} games. Skipped {skipped_count} games.")
        # --- MODIFICATION END ---
        
        return skipped_count

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