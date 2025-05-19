#!/usr/bin/env python3

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Date,
    Float,
    JSON,
    Enum,
    ForeignKey,
    Boolean,
    text
)
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class SportEnum(enum.Enum):
    NFL = 'NFL'
    NBA = 'NBA'
    CFB = 'CFB'
    CBB = 'CBB'


class TeamNameMap(Base):
    """
    Persist ESPN→TeamRankings name mappings so we only fuzzy‐once.
    """
    __tablename__ = 'team_name_map'

    sport     = Column(Enum(SportEnum), primary_key=True)
    espn_name = Column(String,          primary_key=True)
    tr_name   = Column(String,          nullable=False)


class Game(Base):
    __tablename__ = 'games'

    # ESPN IDs can be non-numeric or have leading zeros
    game_id               = Column(String,  primary_key=True)
    date                  = Column(Date,    nullable=False)
    days_since_epoch      = Column(Integer, nullable=False)
    day_of_the_week       = Column(Integer, nullable=False)
    game_time             = Column(String,  nullable=False)
    sport                 = Column(Enum(SportEnum), nullable=False)

    # teams
    team1_id              = Column(Integer, nullable=False)
    team1_name            = Column(String,  nullable=False)
    team1_color           = Column(String,  nullable=True)
    team1_alt_color       = Column(String,  nullable=True)
    team1_logo            = Column(String,  nullable=True)

    team2_id              = Column(Integer, nullable=False)
    team2_name            = Column(String,  nullable=False)
    team2_color           = Column(String,  nullable=True)
    team2_alt_color       = Column(String,  nullable=True)
    team2_logo            = Column(String,  nullable=True)

    # location
    venue_id              = Column(String,  nullable=True)
    city                  = Column(String,  nullable=True)
    state                 = Column(String,  nullable=True)
    country               = Column(String,  nullable=True)
    is_neutral            = Column(Boolean, nullable=False, default=False)
    is_conference         = Column(Boolean, nullable=False, default=False)

    # season breakpoint
    season_type           = Column(Integer, nullable=True)  # 1=pre,2=reg,3=post

    # your combined JSON blob (team/player/weather/misc)
    stats                 = Column(JSON,    nullable=False)
    normalized_stats      = Column(JSON,    nullable=False)

    # pre-game odds
    team1_moneyline       = Column(Float,   nullable=True)
    team2_moneyline       = Column(Float,   nullable=True)
    team1_spread          = Column(Float,   nullable=True)
    team2_spread          = Column(Float,   nullable=True)
    total_score           = Column(Float,   nullable=True)

    # final scores (null until updated)
    team1_score           = Column(Integer, nullable=True)
    team2_score           = Column(Integer, nullable=True)


class Prediction(Base):
    __tablename__ = 'predictions'

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    game_id               = Column(String, ForeignKey('games.game_id'), nullable=False)
    date                  = Column(Date,   nullable=False)
    sport                 = Column(Enum(SportEnum), nullable=False)
    team1_predicted       = Column(Integer, nullable=False)
    team2_predicted       = Column(Integer, nullable=False)
    model_name            = Column(String,  nullable=False)

class ProcessStatus(Base):
    """
    Tracks whether we've run pre- and post-processing for each sport/date.
    """
    __tablename__ = 'process_status'
    sport         = Column(Enum(SportEnum), primary_key=True)
    date          = Column(Date,            primary_key=True)
    preprocessed  = Column(Boolean, default=False, nullable=False)
    postprocessed = Column(Boolean, default=False, nullable=False)

def main():
    engine = create_engine('sqlite:///sports.db', echo=False)
    Base.metadata.create_all(engine)
    # make sure our cleanup‐trigger is in place
    with engine.begin() as conn:
        conn.execute(text("""
        CREATE TRIGGER IF NOT EXISTS reset_status_on_game_delete
        AFTER DELETE ON games
        BEGIN
          UPDATE process_status
          SET preprocessed  = 0,
              postprocessed = 0
          WHERE sport = OLD.sport
            AND date  = OLD.date;
        END;
        """))

    print("✅ Created/updated sports.db with tables + delete-trigger")


if __name__ == '__main__':
    main()
