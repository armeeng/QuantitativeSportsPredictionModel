# team stats
# player stats
# betting odds = 
# weather = some weather API
# final scores = ESPN API

class Pregame:
    def __init__(self, sport: str, date: str):
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

    def get_games_for_date(self):
        """
        Scrape or query an API to get all games scheduled for the specified date and sport.
        Returns:
            List[Dict]: [
                {
                    'id': game_id,
                    'team1_id': id,
                    'team2_id': id,
                    'location': "City or stadium",
                    ...
                }, 
                ...
            ]
        """
        pass

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

    def get_betting_odds(self, team1_id: int, team2_id: int):
        """
        Get betting odds for the game (moneyline, spread, total score).
        Returns:
            Dict: {"team2_moneyline": value, "team2_spread": value, "total_score": value}
        """
        pass

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
