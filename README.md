# ALL

STEPS TO ADD A SPORT:
1. Add to SportsEnum in db_init.py (also need to do something else so the DB knows of this change)
2. Update _ESPN_MAP and _TR_PREFIX
4. Update the following code in Pregame.get_games_for_date
            if dt < now_utc:
                if self.sport in ("CFB", "NFL"):
                    dt -= timedelta(hours=5, minutes=30)
                elif self.sport in ("CBB", "NBA"):
                    dt -= timedelta(hours=4, minutes=30)

5. Update URL list in Pregame.get_team_stats

3. MAKE SURE TEAM NAME MAP IN DB IS GOOD