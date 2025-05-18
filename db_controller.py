#!/usr/bin/env python3
import logging
from datetime import date, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db_init import Base, SportEnum, ProcessStatus
from Pregame import Pregame

# configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

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
    while current >= stop_date:
        print(f"Processing {current}")
        for sport in SportEnum:
            # see if we've already done this sport/date
            rec = session.query(ProcessStatus).get((sport, current))
            if rec is None:
                rec = ProcessStatus(sport=sport, date=current)

            # 1) pregame (stats/odds/weather/etc.)
            if not rec.preprocessed:
                logging.info(f"Preprocessing {sport.value} on {current}")
                pg = Pregame(current, sport.value)
                pg.populate_pregame_data()
                rec.preprocessed = True
                session.merge(rec)
                session.commit()

            # 2) post-process (final scores)
            if not rec.postprocessed:
                logging.info(f"Postprocessing {sport.value} on {current}")
                pg = Pregame(current, sport.value)
                pg.update_final_scores()
                rec.postprocessed = True
                session.merge(rec)
                session.commit()

        current -= timedelta(days=1)

    logging.info("Backfill complete.")

if __name__ == "__main__":
    main()
