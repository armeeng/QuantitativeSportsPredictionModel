#!/usr/bin/env python3
import os
import sys
from datetime import date
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
# Replace 'db_init' with the actual name of your file (e.g., models, db_setup)
from db_init import Prediction, Base

# --- CONFIGURATION ------------------------------------
# Define the path to your database file
DATABASE_URL = "sqlite:///sports.db"
# Define the directory where your trained models are stored
MODELS_DIR = "models/"

# --- DATABASE SETUP ------------------------------------
# Create the SQLAlchemy engine and session factory
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def main():
    """
    Main function to find all models, generate predictions for each,
    and save them to the database.
    """
    # 1. Get a list of all model files from the specified directory
    try:
        model_files = [f for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f))]
    except FileNotFoundError:
        print(f"Error: The models directory was not found at '{MODELS_DIR}'")
        print("Please create it and place your model files inside.")
        return

    if not model_files:
        print(f"No model files found in '{MODELS_DIR}'. Exiting.")
        return

    print(f"Found {len(model_files)} models to process...")
    
    # Get today's date to query for future games
    today_str = date.today().strftime('%Y-%m-%d')
    today_str = '2025-08-10'

    # 2. Loop through each model file to make predictions
    for model_name in model_files:
        print("\n" + "="*60)
        print(f"Processing Model: {model_name}")
        print("="*60)

        # 3. Determine the sport from the filename (e.g., "NFL_..." -> "NFL")
        try:
            sport = model_name.split('_')[0].upper()
            print(f"  Detected Sport: {sport}")
        except IndexError:
            print(f"  Warning: Could not determine sport from filename '{model_name}'. Skipping.")
            continue

        # 4. Dynamically create the prediction query for that sport's future games
        predict_query = (
            f"SELECT * FROM games "
            f"WHERE sport = '{sport}' AND date > '{today_str}' "
            f"ORDER BY date ASC;"
        )
        print(f"  Querying for upcoming '{sport}' games after {today_str}...")

        # 5. Load the model and generate predictions
        try:
            # --- FIX: Remove directory and extension from model name ---
            model_name_stem, _ = os.path.splitext(model_name)
            # Pass just the model name stem (without path/extension) to the MLModel class
            model = MLModel(model_name_stem)
            
            predictions, y_test, test_odds = model.predict(predict_query, mode='prediction')
        except Exception as e:
            print(f"  ERROR: An exception occurred while processing model '{model_name}': {e}")
            continue # Skip to the next model on error

        # 6. Insert or update the predictions in the database
        if y_test is not None:
            # This case handles if the query accidentally included historical games
            print("  Warning: Evaluation data (y_test) was found in prediction mode.")
            print("  This means your query might include games that have already been played.")
            print("  Displaying test results instead of saving to DB for this model.")
            test_evaluator = TestModel(predictions=predictions, y_test=y_test, test_odds=test_odds)
            test_evaluator.display_results()

        elif predictions is not None and len(predictions) > 0:
            print(f"  Generated {len(predictions)} new predictions.")
            db_session = SessionLocal()
            try:
                for pred_data in predictions:
                    existing_pred = db_session.query(Prediction).filter_by(
                        game_id=pred_data['game_id'],
                        # Use the model name stem for DB consistency
                        model_name=model_name_stem
                    ).first()

                    prediction_payload = {
                        'team1_win_prob': pred_data.get('team1_win_prob'),
                        'team1_cover_prob': pred_data.get('team1_cover_prob'),
                        'over_prob': pred_data.get('over_prob'),
                        'pred_team1_score': pred_data.get('pred_team1_score'),
                        'pred_team2_score': pred_data.get('pred_team2_score'),
                    }

                    if existing_pred:
                        # Update existing prediction
                        existing_pred.prediction_data = prediction_payload
                    else:
                        # Create new prediction
                        new_pred_obj = Prediction(
                            game_id=pred_data['game_id'],
                            model_name=model_name_stem,
                            prediction_data=prediction_payload
                        )
                        db_session.add(new_pred_obj)
                
                db_session.commit()
                print(f"  ✅ Successfully saved/updated {len(predictions)} predictions in the database.")

            except Exception as e:
                print(f"  ERROR: A database error occurred: {e}")
                db_session.rollback()
            finally:
                db_session.close()
        else:
            print("  No new games found to predict for this model.")

    print("\n" + "="*60)
    print("Prediction script finished.")
    print("="*60)


if __name__ == "__main__":
    main()

