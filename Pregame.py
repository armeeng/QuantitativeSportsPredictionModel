# team stats
# player stats
# betting odds = 
# weather = some weather API
# final scores = ESPN API

import requests
from datetime import datetime, timezone

class Pregame:

    _ESPN_MAP = {
    'NBA': ('basketball', 'nba'),
    'NFL': ('football', 'nfl'),
    'CFB': ('football', 'college-football'),
    'CBB': ('basketball', 'mens-college-basketball'),
    }
        
    def __init__(self, date, sport: str):
        self.sport = sport
        self.date = date

    # MAIN METHOD TO BE CALLED
    def populate_pregame_data(self):
        games = self.get_games_for_date()
        for game in games:
            team1_stats = self.get_team_stats(game['team1_id'])
            team2_stats = self.get_team_stats(game['team2_id'])
            
            team1_player_stats = self.get_player_stats(game['team1_id'])
            team2_player_stats = self.get_player_stats(game['team2_id'])
            
            betting_odds = self.get_betting_odds(game['team1_id'], game['team2_id'])
            
            weather_info = self.get_weather_info(game['location'])

            combined_stats = {
                "team1": {
                    "team_stats": team1_stats,
                    "player_stats": team1_player_stats
                },
                "team2": {
                    "team_stats": team2_stats,
                    "player_stats": team2_player_stats
                },
                "weather": weather_info,
                "misc": {}  # Placeholder for additional data
            }

            # Insert into pregame_data table
            self.insert_pregame_row(
                game_id=game['id'],
                team1_id=game['team1_id'],
                team2_id=game['team2_id'],
                stats=combined_stats,
                team2_moneyline=betting_odds['team2_moneyline'],
                team2_spread=betting_odds['team2_spread'],
                total_score=betting_odds['total_score']
            )

    # HELPER METHODS BELOW (Logic not implemented, placeholders only)

    def map_team_name(self, espn_name: str) -> str:
        """
        Placeholder method to normalize ESPN's team names to
        whatever naming convention TeamRankings uses.
        Implement your own mapping here.
        """
        # e.g. return TEAM_RANKINGS_MAP.get(espn_name, espn_name)
        return espn_name

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

            # parse the ISO timestamp and compute days since epoch
            dt = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            days_since_epoch = (dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).days

            # pick out home/away teams
            teams = {c["homeAway"]: c for c in comp["competitors"]}
            away = teams["away"]
            home = teams["home"]

            venue = comp.get("venue", {})
            addr  = venue.get("address", {})

            games.append({
                "id":               event["id"],
                "date":             event["date"],                      # full ISO timestamp
                "days_since_epoch": days_since_epoch,
                "game_time":        dt.time().isoformat(),              # HH:MM:SS
                "team1_id":         away["team"]["id"],
                "team1_name":       self.map_team_name(away["team"]["displayName"]),
                "team2_id":         home["team"]["id"],
                "team2_name":       self.map_team_name(home["team"]["displayName"]),
                "venue_id":         venue.get("id"),
                "city":             addr.get("city"),
                "state":            addr.get("state"),
                "country":          addr.get("country"),
                "is_neutral":       comp.get("neutralSite", False),
                "is_conference":    comp.get("conferenceCompetition", False),
            })

        return games
    def get_team_stats(self, team_id: int):
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
