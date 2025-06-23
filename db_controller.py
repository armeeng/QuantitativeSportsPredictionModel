#!/usr/bin/env python3
import logging
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_init import Base, SportEnum, ProcessStatus
from Pregame import Pregame
import sys

for h in logging.root.handlers[:]:
    logging.root.removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    filename="db_controller.log",        # all messages go here
    filemode="a",               # append (use "w" to overwrite each run)
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def main(
    db_url: str = 'sqlite:///sports.db',
    start_date: date = None,
    stop_date:  date = date(2020, 1, 1)
):
    if start_date is None:
        start_date = date.today()

    engine = create_engine(db_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    current = start_date
    current = date(2023, 8, 12)
    while current >= stop_date:
        for sport in SportEnum:
            # see if we've already done this sport/date
            rec = session.get(ProcessStatus, (sport, current))
            if rec is None:
                rec = ProcessStatus(sport=sport, date=current)

            # 1) pregame (stats/odds/weather/etc.)
            if not rec.preprocessed:
                logging.info(f"Preprocessing {sport.value} on {current}")
                pg = Pregame(current, sport.value)
                skipped = pg.populate_pregame_data()
                if skipped == 0:
                    rec.preprocessed = True
                    session.merge(rec)
                    session.commit()

            # 2) post-process (final scores)
            if not rec.postprocessed:
                logging.info(f"Postprocessing {sport.value} on {current}")
                pg = Pregame(current, sport.value)
                skipped = pg.update_final_scores()
                if skipped == 0:
                    rec.postprocessed = True
                    session.merge(rec)
                    session.commit()

        current -= timedelta(days=1)

    logging.info("Backfill complete.")

if __name__ == "__main__":
    main()
