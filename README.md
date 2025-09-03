# Quantitative Sports Prediction Model


## Core Components

- **`db_init.py`**  
  Defines the database schema using SQLAlchemy. It sets up a SQLite database.

- **`Pregame.py`**  
  A data scraping class that fetches game schedules, team stats (from TeamRankings), betting odds (ESPN API), and weather information (Open-Meteo). It processes this data and populates the database.

- **`db_controller.py`**  
  The main script for data ingestion. It systematically iterates through specified dates and sports, calling the `Pregame` class to backfill the database with historical and upcoming game data.

- **`Model.py`**  
  The main machine learning class. It handles feature engineering, training, hyperparameter tuning, and prediction for a wide variety of regression and classification models. It supports different feature strategies (e.g., *flatten* vs. *differential*) and includes model calibration for more accurate probabilities.

- **`TestModel.py`**  
  A detailed evaluation class. After a model predicts outcomes on unseen data, this class calculates performance metrics like accuracy, Profit & Loss (PnL), and statistical significance (p-values). It also runs betting simulations (Monte Carlo and Bootstrap).

- **`ml_train_controller.py`**  
  The script for training a new model. Here, you configure the model type, feature selection, training/testing queries, and other parameters. Running this script trains a model and saves it as a `.joblib` file. It will also print out the performance of the model on unseen data.

- **`ml_predict_controller.py`**  
  The script used to generate predictions for future games. It loads a pre-trained model, queries the database for games that need predictions, and saves the results to the database.

---

## How It Works: The Workflow
The project follows a logical sequence from data collection to prediction and evaluation.

1. **Database Setup**  
   The process begins by creating the SQLite database and its tables.

2. **Data Ingestion**  
   The `db_controller.py` script populates the database. It scrapes pre-game data (stats, odds) first and then runs a post-processing step to add final scores and closing odds once games are complete.

3. **Model Training**  
   With a historical dataset, `ml_train_controller.py` is used to train a model. It splits the data based on a date query, trains the model on the past, and evaluates its performance on the future, printing a detailed report.

4. **Prediction**  
   Once a satisfactory model is trained and saved, `ml_predict_controller.py` loads it to generate predictions for upcoming games that have been added to the database but do not yet have final scores.

---

## Key Features

- **Automated Data Scraping**  
  Gathers comprehensive data from multiple sources (ESPN, TeamRankings, Open-Meteo).

- **Versatile Model Support**  
  Natively supports a wide range of scikit-learn and XGBoost models for both regression (predicting scores) and classification (predicting outcomes).

- **Advanced Evaluation**  
  Goes beyond simple accuracy to evaluate models from a sports betting perspective with PnL calculations, ROI, and simulations.

- **Feature Engineering Options**  
  Easily switch between a simple feature flatten mode and a differential mode that compares team stats directly.

- **Model Calibration**  
  Includes `CalibratedClassifierCV` to ensure that probability outputs from classification models are reliable.

- **Hyperparameter Tuning**  
  Built-in support for `RandomizedSearchCV` to find the best model parameters.

---

# THE REST IS SO I DON'T FORGET IN THE FUTURE. FEEL FREE TO IGNORE.

STEPS TO ADD A SPORT:
1. Add to SportsEnum in db_init.py (DONT FORGET TO RUN IT!!!!)
2. Update _ESPN_MAP and _TR_PREFIX in Pregame.py
4. Update the following code in Pregame.get_games_for_date

     ```python
    if dt < now_utc:
        if self.sport in ("CFB", "NFL"):
            dt -= timedelta(hours=5, minutes=30)
        elif self.sport in ("CBB", "NBA"):
            dt -= timedelta(hours=4, minutes=30)
    ```

5. Update URL list in Pregame.get_team_stats

3. MAKE SURE TEAM NAME MAP IN DB IS GOOD
