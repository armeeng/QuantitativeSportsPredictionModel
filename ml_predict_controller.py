#!/usr/bin/env python3
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# --- Path Correction & Model Imports ---
# This allows the script to find your Model.py, TestModel.py, and database schema files
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Model import MLModel
from TestModel import TestModel
# IMPORTANT: Import your Prediction table class from your database schema file
# Replace 'database_schema' with the actual name of your file (e.g., models, db_setup)
from db_init import Prediction, Base

# --- DATABASE SETUP ------------------------------------
# Define the path to your database file
DATABASE_URL = f"sqlite:///sports.db"

# Create the SQLAlchemy engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def main():
    # ── CONFIG ─────────────────────────────────────
    MODEL_NAME    = "NFL_SP_[430]_lo_nonorm_NFL_date_<_2025_07_10_order_by_date_asc" # This should match the model you want to load

    TEST_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'MLB' AND date > '2025-04-01'"
        "ORDER BY date ASC;"
    )
    
    # Query for games that you want to generate predictions for.
    # These games should NOT have scores in your database yet.
    PREDICT_QUERY = (
        "SELECT * FROM games "
        "WHERE sport = 'NFL' AND date > '2025-08-10'"
        "ORDER BY date ASC;"
    )

    # ── PREDICT --------------------------------────
    print(f"Loading model '{MODEL_NAME}' to make new predictions...")
    model = MLModel(MODEL_NAME)
    
    # The PREDICT_QUERY is parameterized to avoid re-predicting games.
    # We pass the model name to the query execution via pandas' params feature.
    predictions, y_test, test_odds = model.predict(PREDICT_QUERY, mode='prediction')

    # ── OUTPUT & DATABASE INSERTION ─────────────────────────────────────
    if y_test is not None:
        # This block will run if you accidentally query historical data in prediction mode.
        print("Warning: Evaluation data was found, but script is in prediction mode.")
        print("Displaying evaluation results instead of saving to DB.")
        test_evaluator = TestModel(predictions=predictions, y_test=y_test, test_odds=test_odds)
        test_evaluator.display_results()
    elif predictions is not None and len(predictions) > 0:
        # This is the main logic block for inserting or updating predictions.
        print(f"Generated {len(predictions)} new predictions.")

        # Create a new database session
        db_session = SessionLocal()

        try:
            for pred_data in predictions:
                # Check for an existing prediction
                existing_pred = db_session.query(Prediction).filter_by(
                    game_id=pred_data['game_id'],
                    model_name=pred_data['model_name']
                ).first()

                # Prepare the main prediction data payload for the JSON column
                prediction_payload = {
                    'team1_win_prob': pred_data.get('team1_win_prob'),
                    'team1_cover_prob': pred_data.get('team1_cover_prob'),
                    'over_prob': pred_data.get('over_prob'),
                    'pred_team1_score': pred_data.get('pred_team1_score'),
                    'pred_team2_score': pred_data.get('pred_team2_score'),
                }

                if existing_pred:
                    # If prediction exists, update its data
                    print(f"Updating prediction for game_id: {pred_data['game_id']}")
                    existing_pred.prediction_data = prediction_payload
                else:
                    # If prediction does not exist, create a new one
                    print(f"Creating new prediction for game_id: {pred_data['game_id']}")
                    new_pred_obj = Prediction(
                        game_id=pred_data['game_id'],
                        model_name=pred_data['model_name'],
                        prediction_data=prediction_payload
                    )
                    db_session.add(new_pred_obj)

            # Commit the transaction to save all changes (updates and inserts)
            db_session.commit()
            print(f"Successfully saved/updated {len(predictions)} predictions in the database.")

        except Exception as e:
            print(f"An error occurred during database operation: {e}")
            db_session.rollback() # Roll back the transaction on error
        finally:
            db_session.close() # Always close the session

    else:
        print("No new games to predict or an error occurred.")

if __name__ == "__main__":
    main()