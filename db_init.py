#!/usr/bin/env python3

from sqlalchemy import (
    create_engine, Column, Integer, String, Date, Float, JSON, Enum
)
from sqlalchemy.ext.declarative import declarative_base
import enum

Base = declarative_base()

class SportEnum(enum.Enum):
    NFL = 'NFL'
    NBA = 'NBA'
    CFB = 'CFB'
    CBB = 'CBB'

class Game(Base):
    __tablename__ = 'games'
    game_id          = Column(Integer, primary_key=True)
    date             = Column(Date,   nullable=False)
    sport            = Column(Enum(SportEnum), nullable=False)
    team1            = Column(String, nullable=False)
    team2            = Column(String, nullable=False)
    stats            = Column(JSON,   nullable=False)   # your combined stats array
    team1_moneyline  = Column(Float)
    team2_moneyline  = Column(Float)
    team1_spread     = Column(Float)
    team2_spread     = Column(Float)
    total_score      = Column(Float)
    team1_score      = Column(Integer, nullable=True)   # null until updated
    team2_score      = Column(Integer, nullable=True)

class Prediction(Base):
    __tablename__ = 'predictions'
    id               = Column(Integer, primary_key=True, autoincrement=True)
    game_id          = Column(Integer, nullable=False)
    date             = Column(Date, nullable=False)
    sport            = Column(Enum(SportEnum), nullable=False)
    team1_predicted  = Column(Integer, nullable=False)
    team2_predicted  = Column(Integer, nullable=False)
    model_name       = Column(String,  nullable=False)

def main():
    # Creates (or opens) sports.db in your CWD
    engine = create_engine('sqlite:///sports.db')
    Base.metadata.create_all(engine)
    print("✅ Created database 'sports.db' with tables: games, predictions")

if __name__ == '__main__':
    main()
