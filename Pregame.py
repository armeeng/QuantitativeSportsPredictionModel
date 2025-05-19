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
                    total_score
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?,
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

                odds.get("team1_moneyline"),
                odds.get("team2_moneyline"),
                odds.get("team1_spread"),
                odds.get("team2_spread"),
                odds.get("total_score"),
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
            logging.error(f"Unrecognized stat value: {text!r}")
            return 0.0

    def get_team_stats(self, team_name: str):
        """
        For each TeamRankings stat page for this.sport, scrape the first table,
        find the row matching team_name, and return both raw and normalized stats.

        Returns:
            Dict[str, Dict[str, Dict[str, str|float]]]:
              {
                "<stat-slug>": {
                  "raw":        { "Rank": "5",   "2024": "28.9", … },
                  "normalized": { "Rank": 0.04,  "2024": 0.83,  … }
                },
                …
              }
        If a table is missing, raises RuntimeError.
        If team_name is not found in any page, logs and returns None.
        """
        url_lists = {
            'CBB': [
                #"https://www.teamrankings.com/ncaa-basketball/stat/points-per-game",
                "https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other/"
                # …
            ],
            'CFB': [
                "https://www.teamrankings.com/college-football/stat/points-per-game",
                # …
            ],
            'NBA': [
                "https://www.teamrankings.com/nba/stat/points-per-game",
                # …
            ],
            'NFL': [
                "https://www.teamrankings.com/nfl/stat/points-per-game",
                # …
            ],
            'MLB': [
                "https://www.teamrankings.com/mlb/stat/runs-per-game",
                # …
            ],
        }

        if self.sport not in url_lists:
            raise ValueError(f"No stat pages for {self.sport}")

        all_stats = {}
        for url in url_lists[self.sport]:
            dated_url = f"{url}?date={self.date.isoformat()}"
            resp = requests.get(dated_url, headers={"User-Agent":"Mozilla/5.0"})
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                raise RuntimeError(f"No table found at {dated_url}")

            # headers: ["Rank","Team","2024",...]
            headers = [th.get_text(strip=True) for th in table.select("thead th")]

            raw_rows = []
            num_rows = []
            for tr in table.select("tbody tr"):
                tds = tr.select("td")
                raw = {}
                # build raw, but for "Team" pull only the <a> text
                for i, h in enumerate(headers):
                    cell = tds[i]
                    if h == "Team":
                        a = cell.find("a")
                        raw[h] = a.get_text(strip=True) if a else cell.get_text(strip=True)
                    else:
                        raw[h] = cell.get_text(strip=True)
                raw_rows.append(raw)

                # numeric parse for non-Team
                nums = { h: self._parse_stat_value(raw[h])
                         for h in headers if h != "Team" }
                num_rows.append(nums)

            # compute min/max per column
            min_max = {}
            for h in num_rows[0]:
                col = [r[h] for r in num_rows]
                min_max[h] = (min(col), max(col))

            # find our team
            for i, raw in enumerate(raw_rows):
                if raw["Team"] == team_name:
                    raw_stats = {
                        h: self._parse_stat_value(raw[h])
                        for h in headers
                        if h != "Team"
                    }
                    norm_stats = {}
                    for h, val in num_rows[i].items():
                        lo, hi = min_max[h]
                        norm_stats[h] = (val - lo) / (hi - lo) if hi > lo else 0.0

                    parsed = urlparse(url)
                    slug = parsed.path.rstrip("/").split("/")[-1]
                    all_stats[slug] = {"raw": raw_stats, "normalized": norm_stats}
                    break
            else:
                logging.error(f"Team {team_name!r} not found at {dated_url!r}")
                return None

        return all_stats

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
            if not results:
                logging.error(f"Geocoding failed for {city}, {state}, {country}")
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

        # 3) Choose endpoint
        if dt < now_utc:
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

    def update_final_scores(self):
        """
        Scrape ESPN's scoreboard for self.date/self.sport, and update
        team1_score/team2_score in the games table for each completed game.
        """
        # 1) Build ESPN API URL
        try:
            category, league = self._ESPN_MAP[self.sport]
        except KeyError:
            raise ValueError(f"Unsupported sport code: {self.sport!r}")

        url = (
            f"https://site.api.espn.com/apis/site/v2/"
            f"sports/{category}/{league}/scoreboard"
        )
        date_str = self.date.strftime("%Y%m%d")

        # 2) Fetch scoreboard
        resp = requests.get(url, params={"dates": date_str})
        resp.raise_for_status()
        data = resp.json()

        # 3) Loop through events
        updated = 0
        skip = 0
        for event in data.get("events", []):
            comp = event["competitions"][0]
            status = comp.get("status", {}).get("type", {})
            # only update if game is completed
            if not status.get("completed", False):
                continue

            # map away=team1, home=team2
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away = teams.get("away")
            home = teams.get("home")

            # parse scores (may come as strings)
            try:
                team1_score = int(away.get("score", 0))
                team2_score = int(home.get("score", 0))
            except (TypeError, ValueError):
                logging.error(f"Invalid score for game {event['id']}: "
                              f"{away.get('score')} vs {home.get('score')}")
                continue

            # 4) Update DB
            cur = self.conn.execute(
                "UPDATE games "
                "SET team1_score = ?, team2_score = ? "
                "WHERE game_id = ?",
                (team1_score, team2_score, event["id"])
            )
            if cur.rowcount == 0:
                skip += 1
                logging.warning(f"No local game found for ESPN ID {event['id']}")
            else:
                updated += 1

        # 5) Commit once at the end
        self.conn.commit()
        logging.info(f"Updated final scores for {updated} games. Skipped {skip} games.")
        return skip

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