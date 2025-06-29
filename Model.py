import os
import json
import sqlite3
import glob
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import randint, uniform


from TestModel import TestModel

pd.set_option('display.max_colwidth', None)

# A simple base class for defining common attributes.
class BaseModel:
    def __init__(self, model_name: str, column: str):
        self.model_name = model_name
        self.column = column

class MLModel(BaseModel):
    """
    A simplified ML class focused strictly on training models and generating predictions.

    This class trains a model on historical sports data from a training query and
    evaluates it on a separate test query. All metric calculations are handled
    by the TestModel class. The test set predictions, true outcomes, and betting
    odds are stored as instance attributes for external evaluation.
    It supports training on a full or random subset of features and uses StandardScaler
    to scale features.
    """

    _HYPERPARAMETER_GRIDS = {
        'linear_regression': {},
        'random_forest_regressor': {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(range(10, 111, 20)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2', None]
        },
        'xgboost_regressor': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5)
        },
        'mlp_regressor': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': uniform(0.0001, 0.01),
            'learning_rate_init': uniform(0.001, 0.1),
        },
        'knn_regressor': {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'svr': { # Note: Parameters are prefixed with 'estimator__' for MultiOutputRegressor
            'estimator__C': uniform(0.1, 10),
            'estimator__gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5)),
            'estimator__kernel': ['rbf', 'poly', 'sigmoid']
        },
        'logistic_regression': {
            'C': uniform(0.1, 10),
            'penalty': ['l1', 'l2']
        },
        'knn_classifier': {
            'n_neighbors': randint(3, 20),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        },
        'svc': {
            'C': uniform(0.1, 10),
            'gamma': ['scale', 'auto'] + list(uniform(0.001, 0.1).rvs(5)),
            'kernel': ['rbf', 'poly', 'sigmoid']
        },
        'random_forest_classifier': {
            'n_estimators': randint(50, 500),
            'max_depth': [None] + list(range(10, 111, 20)),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'xgboost_classifier': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5)
        },
        'mlp_classifier': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'sgd'],
            'alpha': uniform(0.0001, 0.01),
            'learning_rate_init': uniform(0.001, 0.1),
        },
        'gradient_boosting_classifier': {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'max_depth': randint(3, 8),
            'subsample': uniform(0.7, 0.3)
        },
        'gaussian_nb': {}
    }
    _HYPERPARAMETER_GRIDS['random_forest'] = _HYPERPARAMETER_GRIDS['random_forest_regressor']
    _HYPERPARAMETER_GRIDS['xgboost'] = _HYPERPARAMETER_GRIDS['xgboost_regressor']
    _HYPERPARAMETER_GRIDS['mlp'] = _HYPERPARAMETER_GRIDS['mlp_regressor']
    _HYPERPARAMETER_GRIDS['neural_network'] = _HYPERPARAMETER_GRIDS['mlp_regressor']
    _HYPERPARAMETER_GRIDS['svm'] = _HYPERPARAMETER_GRIDS['svc']

    _CLASSIFIER_TYPES = {
        'logistic_regression', 'knn_classifier', 'svc', 'random_forest_classifier',
        'xgboost_classifier', 'mlp_classifier', 'gradient_boosting_classifier', 'gaussian_nb'
    }

    _DEFAULT_CATEGORICAL_FEATURES = [
        "team1_id", "team2_id", "venue_id", "season_type", "day", "month",
        "year", "day_of_week"
    ]

    def __init__(self, model_name: str, model_type: str = 'linear_regression',
                 column: str = "normalized_stats",
                 # NEW: Parameter to control how numerical features are created
                 feature_engineering_mode: str = 'flatten',
                 # Granular feature selection parameters
                 numerical_feature_indices: list[int] = None,
                 categorical_feature_names: list[str] = None,
                 include_market_spread: bool = False,
                 include_market_total: bool = False,
                 # Random subset selection
                 use_random_subset_of_numerical_features: bool = False,
                 subset_fraction: float = None,
                 # Hyperparameter tuning and reproducibility
                 hyperparameter_tuning: bool = False, tuning_n_iter: int = 50, tuning_cv: int = 5,
                 random_state: int = 42):
        """
        REWRITTEN: `__init__` now accepts a `feature_engineering_mode` parameter.

        Args:
            feature_engineering_mode (str): How to process numerical features.
                'flatten': (Default) Flattens all numerical stats into one vector.
                'differential': Creates features from team2_stats - team1_stats and appends other numericals.
            ... (other args)
        """
        super().__init__(model_name, column=column)
        self.random_state = random_state

        # NEW: Validate and store the feature engineering mode
        valid_modes = ['flatten', 'differential']
        if feature_engineering_mode not in valid_modes:
            raise ValueError(f"Invalid feature_engineering_mode. Must be one of {valid_modes}")
        self.feature_engineering_mode = feature_engineering_mode

        self._MODELS = {
            # Regressors
            'linear_regression': LinearRegression,
            'random_forest_regressor': lambda: RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'xgboost_regressor': lambda: XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=self.random_state),
            'mlp_regressor': lambda: MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state),
            'knn_regressor': lambda: KNeighborsRegressor(n_neighbors=5),
            'svr': lambda: MultiOutputRegressor(SVR(kernel='rbf')),

            # Classifiers
            'logistic_regression': lambda: LogisticRegression(solver='liblinear', max_iter=1000, random_state=self.random_state),
            'knn_classifier': lambda: KNeighborsClassifier(n_neighbors=5),
            'svc': lambda: SVC(probability=True, random_state=self.random_state),
            'random_forest_classifier': lambda: RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'xgboost_classifier': lambda: XGBClassifier(objective='binary:logistic', n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=self.random_state),
            'mlp_classifier': lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.random_state),
            'gradient_boosting_classifier': lambda: GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'gaussian_nb': lambda: GaussianNB(),
        }
        # Legacy aliases
        self._MODELS['random_forest'] = self._MODELS['random_forest_regressor']
        self._MODELS['xgboost'] = self._MODELS['xgboost_regressor']
        self._MODELS['mlp'] = self._MODELS['mlp_regressor']
        self._MODELS['neural_network'] = self._MODELS['mlp_regressor']
        self._MODELS['svm'] = self._MODELS['svc']

        self.model_type = model_type.lower()
        if self.model_type not in self._MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}. Supported types are {list(self._MODELS.keys())}")

        if use_random_subset_of_numerical_features and numerical_feature_indices is not None:
            raise ValueError("Cannot set `use_random_subset_of_numerical_features` to True and provide `numerical_feature_indices` simultaneously.")

        # Store feature selection parameters
        self.numerical_feature_indices = numerical_feature_indices
        self.categorical_feature_names = categorical_feature_names
        self.include_market_spread = include_market_spread
        self.include_market_total = include_market_total
        self.use_random_subset_of_numerical_features = use_random_subset_of_numerical_features

        if subset_fraction is None: self.subset_fraction = random.uniform(0.01, 1.0)
        else:
            if not (0 < subset_fraction <= 1.0): raise ValueError("If specified, subset_fraction must be > 0 and <= 1.")
            self.subset_fraction = subset_fraction

        self.hyperparameter_tuning = hyperparameter_tuning
        self.tuning_n_iter = tuning_n_iter
        self.tuning_cv = tuning_cv

        # Instance attributes to be populated during training
        self.predictions = None
        self.y_test = None
        self.test_odds = None
        self.scaler = None
        self.one_hot_encoder = None
        self.model_ = None
        
        # Final feature names and indices used for training the model
        self.feature_names_ = None
        self.trained_numerical_indices_ = None
        self.final_numerical_feature_names_ = None

    # =================================================================================
    # Public API: Main Methods
    # =================================================================================

    def train(self, train_query: str, test_query: str):
        """
        Trains a model using data from `train_query` and evaluates it on `test_query`.

        Args:
            train_query (str): The SQL query to fetch the training dataset.
            test_query (str): The SQL query to fetch the test dataset for evaluation.
        """
        if self.model_type in self._CLASSIFIER_TYPES:
            self._train_classifier(train_query, test_query)
        else:
            self._train_regressor(train_query, test_query)

    def predict(self, query: str) -> list:
        """
        REWRITTEN: Loads the latest model and metadata, then processes new data using the
        exact same feature engineering pipeline from training to generate predictions.
        """
        try:
            info = self._load_model()
            model = info['model']
            self.scaler = info.get('scaler')
            self.one_hot_encoder = info.get('one_hot_encoder')
            self.column = info['column']
            # Load the feature selection metadata the model was trained with
            self.trained_numerical_indices_ = info.get('trained_numerical_indices')
            self.categorical_feature_names = info.get('categorical_feature_names')
            # NEW: Load the feature engineering mode
            self.feature_engineering_mode = info.get('feature_engineering_mode', 'flatten') # Default to flatten for old models

            # Handle backward compatibility for market feature flags
            old_market_flag = info.get('include_market_features', False)
            self.include_market_spread = info.get('include_market_spread', old_market_flag)
            self.include_market_total = info.get('include_market_total', old_market_flag)

        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []

        df = self._load_data_from_db(query)
        if df.empty:
            print("Query returned no data to predict on.")
            return []

        # --- Start Prediction Feature Engineering Pipeline ---
        # This pipeline mirrors the training process exactly.
        
        # 1. Determine which categorical features to use (from loaded model info)
        cats_to_use = self.categorical_feature_names if self.categorical_feature_names is not None else self._DEFAULT_CATEGORICAL_FEATURES

        # 2. Prepare numerical features based on the saved mode
        X_num, _ = self._prepare_numerical_features(df)

        # 3. Select the specific numerical features the model was trained on
        X_num_selected = X_num[:, self.trained_numerical_indices_] if self.trained_numerical_indices_ is not None else X_num
        
        # 4. Prepare categorical features
        cat_df = self._extract_categorical_features(df, features_to_extract=cats_to_use)
        X_cat = self.one_hot_encoder.transform(cat_df) if self.one_hot_encoder and cat_df.shape[1] > 0 else np.array([[] for _ in range(len(df))])

        # 5. Combine numerical and categorical
        X_combined = np.hstack([X_num_selected, X_cat])
        
        # 6. Conditionally add market features if the model was trained with them
        X_final = X_combined
        if self.include_market_spread:
            spread_feature = pd.to_numeric(df["team1_spread"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_final = np.hstack((X_final, spread_feature))

        if self.include_market_total:
            total_feature = pd.to_numeric(df["total_score"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_final = np.hstack((X_final, total_feature))
        # --- End Prediction Feature Engineering Pipeline ---

        # 7. Scale and Predict
        if self.scaler:
            X_scaled = self.scaler.transform(X_final)
        else:
            print("Warning: No scaler found. Predicting on unscaled data.")
            X_scaled = X_final

        if isinstance(model, dict):  # Classifier
            prob_win = model['win'].predict_proba(X_scaled)[:, 1]
            prob_cover = model['spread'].predict_proba(X_scaled)[:, 1]
            prob_over = model['over'].predict_proba(X_scaled)[:, 1]
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "team1_win_prob": pw, "team1_cover_prob": pc, "over_prob": po}
                for gid, t1, t2, pw, pc, po in zip(df["game_id"], df["team1_id"], df["team2_id"], prob_win, prob_cover, prob_over)
            ]
        else:  # Regressor
            predictions = model.predict(X_scaled)
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "pred_team1": p1, "pred_team2": p2}
                for gid, t1, t2, (p1, p2) in zip(df["game_id"], df["team1_id"], df["team2_id"], predictions)
            ]

    def get_feature_importance(self, model=None, X_test=None, y_test=None, n_top_features=20):
        """
        REWRITTEN: Calculates and returns feature importances, filtering for ONLY
        numerical features and adding a column for their original index.
        """
        if model is None: model = self.model_
        if self.feature_names_ is None:
            print("Feature names are not available. Please train the model first.")
            return None

        # --- Calculate full list of importances ---
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1: importances = np.mean(np.abs(model.coef_), axis=0)
            else: importances = np.abs(model.coef_)
        elif X_test is not None and y_test is not None:
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=self.random_state, n_jobs=-1)
            importances = result.importances_mean
        else:
            print(f"Cannot get feature importance for model type {type(model).__name__} without test data.")
            return None

        # --- Create, Filter, and Enhance the Importance DataFrame ---
        full_importance_df = pd.DataFrame({
            'importance': importances,
            'feature': self.feature_names_
        })

        # Filter for only the numerical features used in the model
        if not self.final_numerical_feature_names_:
            print("Warning: No numerical feature names found to calculate importance for.")
            return pd.DataFrame(columns=['feature_index', 'feature', 'importance'])

        numerical_importance_df = full_importance_df[
            full_importance_df['feature'].isin(self.final_numerical_feature_names_)
        ].copy()

        # Determine the original indices for the numerical features
        if self.trained_numerical_indices_ is not None:
            indices = self.trained_numerical_indices_
        else:  # This case occurs if all numerical features were used
            indices = list(range(len(self.final_numerical_feature_names_)))
        
        # Add the original index column
        if len(numerical_importance_df) == len(indices):
            numerical_importance_df['feature_index'] = indices
        else:
            print("Warning: Mismatch between number of numerical features and indices. Cannot add 'feature_index' column.")
            numerical_importance_df['feature_index'] = pd.NA

        # Select and reorder columns, then sort by importance
        final_df = numerical_importance_df[['feature_index', 'feature', 'importance']]
        return final_df.sort_values(by='importance', ascending=False)

    # =================================================================================
    # Internal Training Methods
    # =================================================================================

    def _train_regressor(self, train_query: str, test_query: str):
        """ REWRITTEN: Trains a regression model using the new granular feature engineering pipeline."""
        # Load Data
        df_train = self._load_data_from_db(train_query)
        df_test = self._load_data_from_db(test_query)

        df_train.dropna(subset=["team1_score", "team2_score"], inplace=True)
        df_test.dropna(subset=["team1_score", "team2_score"], inplace=True)

        if df_train.empty or df_test.empty:
            print("Training or test query returned no data. Aborting.")
            return

        # Prepare features using the new modular pipeline
        X_train, X_test = self._prepare_and_select_features(df_train, df_test)
        y_train = df_train[["team1_score", "team2_score"]].to_numpy()
        self.y_test = df_test[["team1_score", "team2_score"]].to_numpy()
        
        # Save training data for inspection
        self._save_training_dataframe(X_train, self.feature_names_, f"{self.model_name}_training_data")

        self.test_odds = self._extract_test_odds(df_test)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Model
        initial_model = self._get_model()
        if self.hyperparameter_tuning:
            print(f"--- Starting Hyperparameter Tuning for {self.model_type} ---")
            self.model_ = self._tune_hyperparameters(initial_model, X_train_scaled, y_train)
            print("--- Hyperparameter Tuning Finished ---")
            print(f"Best model parameters: {self.model_.get_params()}")
        else:
            print("--- Training with default hyperparameters ---")
            self.model_ = initial_model
            self.model_.fit(X_train_scaled, y_train)

        # Evaluate Model
        self.predictions = self.model_.predict(X_test_scaled)
        test_evaluator = TestModel(predictions=self.predictions, y_test=self.y_test, test_odds=self.test_odds)
        test_evaluator.display_results()

        # Feature Importance
        feature_importance = self.get_feature_importance(model=self.model_, X_test=X_test_scaled, y_test=self.y_test)
        if feature_importance is not None:
            os.makedirs("Feature Importance", exist_ok=True)
            feature_importance.to_csv("Feature Importance/feature_importance.csv", index=False)
            print("Saved feature importance to feature_importance.csv")

        self._save_model(self.model_, self.scaler, train_query)
        print(f"Trained {self.model_name} ({self.model_type}) on {len(X_train)} train / {len(X_test)} test games.")

    def _train_classifier(self, train_query: str, test_query: str):
        """ REWRITTEN: Trains classification models using the new granular feature engineering pipeline."""
        # Load Data
        df_train = self._load_data_from_db(train_query)
        df_test = self._load_data_from_db(test_query)

        df_train.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)
        df_test.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)

        if df_train.empty or df_test.empty:
            print("Training or test query returned no data. Aborting.")
            return

        # Prepare features using the new modular pipeline
        X_train, X_test = self._prepare_and_select_features(df_train, df_test)

        # Save training data for inspection
        self._save_training_dataframe(X_train, self.feature_names_, f"{self.model_name}_training_data")
        
        # Prepare Target Variables (y)
        y_train_win = (df_train["team1_score"] > df_train["team2_score"]).astype(int)
        y_train_spread_outcome = (df_train["team1_score"] - df_train["team2_score"]) + pd.to_numeric(df_train["team1_spread"], errors='coerce').fillna(0)
        y_train_total_outcome = (df_train["team1_score"] + df_train["team2_score"]) - pd.to_numeric(df_train["total_score"], errors='coerce').fillna(0)
        
        self.y_test = df_test[["team1_score", "team2_score"]]
        y_test_win = (df_test["team1_score"] > df_test["team2_score"]).astype(int)
        y_test_spread_outcome = (df_test["team1_score"] - df_test["team2_score"]) + pd.to_numeric(df_test["team1_spread"], errors='coerce').fillna(0)
        y_test_total_outcome = (df_test["team1_score"] + df_test["team2_score"]) - pd.to_numeric(df_test["total_score"], errors='coerce').fillna(0)
        
        self.test_odds = self._extract_test_odds(df_test)

        # Scale Features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Individual Models (Win, Spread, Over/Under)
        model_win = self._train_single_classifier("WIN", X_train_scaled, y_train_win)
        
        train_spread_non_push = y_train_spread_outcome != 0
        model_spread = self._train_single_classifier("SPREAD", X_train_scaled[train_spread_non_push], (y_train_spread_outcome[train_spread_non_push] > 0).astype(int))

        train_total_non_push = y_train_total_outcome != 0
        model_over = self._train_single_classifier("OVER/UNDER", X_train_scaled[train_total_non_push], (y_train_total_outcome[train_total_non_push] > 0).astype(int))

        print("\n--- Model Training Finished ---\n")
        self.model_ = {'win': model_win, 'spread': model_spread, 'over': model_over}
        
        # Evaluation and Saving
        self.predictions = {
            'win': self.model_['win'].predict_proba(X_test_scaled),
            'spread': self.model_['spread'].predict_proba(X_test_scaled),
            'over': self.model_['over'].predict_proba(X_test_scaled)
        }
        test_evaluator = TestModel(predictions=self.predictions, y_test=self.y_test, test_odds=self.test_odds)
        test_evaluator.display_results()
        
        # Feature Importance
        os.makedirs("Feature Importance", exist_ok=True)
        df_win = self.get_feature_importance(model=model_win, X_test=X_test_scaled, y_test=y_test_win)
        df_win.to_csv("Feature Importance/feature_importance_win.csv", index=False)
        print("\nSaved Win model importances to feature_importance_win.csv")

        mask = y_test_spread_outcome != 0
        df_spread = self.get_feature_importance(model=model_spread, X_test=X_test_scaled[mask], y_test=(y_test_spread_outcome[mask] > 0).astype(int))
        df_spread.to_csv("Feature Importance/feature_importance_spread.csv", index=False)
        print("Saved Spread model importances to feature_importance_spread.csv")

        mask = y_test_total_outcome != 0
        df_over = self.get_feature_importance(model=model_over, X_test=X_test_scaled[mask], y_test=(y_test_total_outcome[mask] > 0).astype(int))
        df_over.to_csv("Feature Importance/feature_importance_overunder.csv", index=False)
        print("Saved Over/Under model importances to feature_importance_overunder.csv")

        self._save_model(self.model_, self.scaler, train_query)
        print(f"Trained {self.model_name} ({self.model_type}) with {len(X_train)} train / {len(X_test)} test games.")

    def _train_single_classifier(self, name: str, X_train, y_train):
        """Helper to tune or train a single classification model."""
        initial_model = self._get_model()
        if self.hyperparameter_tuning:
            print(f"\n--- Tuning {name} model ({self.model_type}) ---")
            model = self._tune_hyperparameters(initial_model, X_train, y_train)
            print(f"Best {name} model parameters: {model.get_params()}")
        else:
            model = initial_model.fit(X_train, y_train)
        return model

    # =================================================================================
    # Internal Helper Methods: Feature Engineering
    # =================================================================================

    def _prepare_and_select_features(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        NEW: A master pipeline for feature creation and selection.
        This orchestrates the numerical, categorical, and market feature steps.
        """
        # 1. Determine which categorical features to use from instance settings
        cats_to_use = self.categorical_feature_names if self.categorical_feature_names is not None else self._DEFAULT_CATEGORICAL_FEATURES

        # 2. Prepare numerical features from the JSON column based on the chosen mode
        print(f"INFO: Using '{self.feature_engineering_mode}' mode for numerical features.")
        X_train_num, X_test_num, all_num_feature_names = self._prepare_numerical_features(df_train, df_test)

        # 3. Prepare one-hot encoded categorical features
        X_train_cat, X_test_cat, cat_feature_names = self._prepare_categorical_features(df_train, df_test, cats_to_use)

        # 4. Select a subset of numerical features based on instance settings
        X_train_num_final, X_test_num_final, final_num_names, self.trained_numerical_indices_ = self._select_numerical_features(
            X_train_num, X_test_num, all_num_feature_names
        )
        self.final_numerical_feature_names_ = final_num_names

        # 5. Combine the selected numerical features with the categorical features
        X_train_combined = np.hstack([X_train_num_final, X_train_cat])
        X_test_combined = np.hstack([X_test_num_final, X_test_cat])
        
        # Store the names of the features combined so far
        self.feature_names_ = self.final_numerical_feature_names_ + cat_feature_names

        # 6. Conditionally add market features
        X_train_final, X_test_final = X_train_combined, X_test_combined

        if self.include_market_spread:
            print("INFO: Including market spread as a feature.")
            spread_train = pd.to_numeric(df_train["team1_spread"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_train_final = np.hstack((X_train_final, spread_train))

            spread_test = pd.to_numeric(df_test["team1_spread"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_test_final = np.hstack((X_test_final, spread_test))
            
            self.feature_names_.append('market_team1_spread')

        if self.include_market_total:
            print("INFO: Including market total score as a feature.")
            total_train = pd.to_numeric(df_train["total_score"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_train_final = np.hstack((X_train_final, total_train))

            total_test = pd.to_numeric(df_test["total_score"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
            X_test_final = np.hstack((X_test_final, total_test))

            self.feature_names_.append('market_total_score')

        print(f"INFO: Final feature count is {X_train_final.shape[1]}.")
        return X_train_final, X_test_final

    def _prepare_numerical_features(self, *dataframes: pd.DataFrame) -> tuple:
        """
        MODIFIED: Routes to the correct feature engineering logic based on self.feature_engineering_mode.
        """
        if self.feature_engineering_mode == 'differential':
            return self._prepare_differential_features(*dataframes)
        else: # Default to 'flatten'
            return self._prepare_flattened_features(*dataframes)

    def _prepare_flattened_features(self, *dataframes: pd.DataFrame) -> tuple:
        """
        Original logic to flatten all numerical features from the JSON blob.
        """
        if not dataframes: return ()

        # Determine master feature names from the first available valid JSON object
        master_feature_names = None
        all_parsed_json = [df[self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x) for df in dataframes]

        for parsed_json in all_parsed_json:
            for js_obj in parsed_json:
                if js_obj:
                    temp_obj = {k: v for k, v in js_obj.items() if k not in self._DEFAULT_CATEGORICAL_FEATURES}
                    _, discovered_names = self._flatten_json_to_list(temp_obj)
                    if discovered_names:
                        master_feature_names = discovered_names
                        break
            if master_feature_names: break
        
        if not master_feature_names:
            print("Warning: Could not determine numerical feature names from any dataframe. Returning empty arrays.")
            return (np.array([[] for _ in range(len(df))]) for df in dataframes) + ([],)
            
        name_to_index = {name: i for i, name in enumerate(master_feature_names)}

        results = []
        for parsed_json in all_parsed_json:
            feature_list = []
            for js_obj in parsed_json:
                row_values_ordered = [0.0] * len(master_feature_names)
                if js_obj:
                    temp_obj = {k: v for k, v in js_obj.items() if k not in self._DEFAULT_CATEGORICAL_FEATURES}
                    row_values_raw, row_names_raw = self._flatten_json_to_list(temp_obj)
                    for name, value in zip(row_names_raw, row_values_raw):
                        if name in name_to_index:
                            row_values_ordered[name_to_index[name]] = value
                feature_list.append(row_values_ordered)
            results.append(np.array(feature_list, dtype=float))

        return tuple(results) + (master_feature_names,)

    def _prepare_differential_features(self, *dataframes: pd.DataFrame) -> tuple:
        """
        NEW: Creates features by calculating team2_stats - team1_stats and appending other numericals.
        """
        if not dataframes: return ()

        # Determine feature names from the first valid JSON object
        diff_names, other_names = None, None
        all_parsed_json = [df[self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x) for df in dataframes]
        
        for parsed_json in all_parsed_json:
            for js_obj in parsed_json:
                if js_obj and 'team1_stats' in js_obj:
                    # Get differential names from team1_stats
                    _, stat_names = self._flatten_json_to_list(js_obj['team1_stats'])
                    diff_names = [f"diff_{name}" for name in stat_names]

                    # Get names of other numerical features
                    other_obj = {k: v for k, v in js_obj.items() if k not in self._DEFAULT_CATEGORICAL_FEATURES and k not in ['team1_stats', 'team2_stats']}
                    _, other_names = self._flatten_json_to_list(other_obj)
                    if diff_names: break
            if diff_names: break

        if not diff_names:
            print("Warning: Could not find 'team1_stats' to create differential features. Falling back to flatten mode.")
            return self._prepare_flattened_features(*dataframes)

        master_feature_names = diff_names + other_names
        
        # Process all dataframes
        results = []
        for parsed_json in all_parsed_json:
            feature_list = []
            for js_obj in parsed_json:
                # Initialize with zeros
                t1_vals, t2_vals, other_vals = [0.0] * len(diff_names), [0.0] * len(diff_names), [0.0] * len(other_names)
                
                if js_obj:
                    # Get team stats
                    if 'team1_stats' in js_obj:
                        t1_vals, _ = self._flatten_json_to_list(js_obj['team1_stats'])
                    if 'team2_stats' in js_obj:
                        t2_vals, _ = self._flatten_json_to_list(js_obj['team2_stats'])
                    
                    # Get other numerical stats
                    other_obj = {k: v for k, v in js_obj.items() if k not in self._DEFAULT_CATEGORICAL_FEATURES and k not in ['team1_stats', 'team2_stats']}
                    other_vals, _ = self._flatten_json_to_list(other_obj)

                # Ensure lists have correct length before operations
                t1_vals = (t1_vals + [0.0] * len(diff_names))[:len(diff_names)]
                t2_vals = (t2_vals + [0.0] * len(diff_names))[:len(diff_names)]
                other_vals = (other_vals + [0.0] * len(other_names))[:len(other_names)]

                # Calculate differential features
                diff_features = np.array(t2_vals) - np.array(t1_vals)
                
                # Combine into a single row
                full_row = np.concatenate([diff_features, np.array(other_vals)])
                feature_list.append(full_row)
            
            results.append(np.array(feature_list, dtype=float))

        return tuple(results) + (master_feature_names,)


    def _prepare_categorical_features(self, df_train, df_test, cats_to_use):
        """
        NEW: Extracts, one-hot encodes, and returns categorical features.
        """
        if not cats_to_use:
            print("INFO: No categorical features selected.")
            empty_array_train = np.empty((len(df_train), 0))
            empty_array_test = np.empty((len(df_test), 0))
            return empty_array_train, empty_array_test, []

        print(f"--- Encoding categorical features: {cats_to_use} ---")
        cat_df_train = self._extract_categorical_features(df_train, features_to_extract=cats_to_use)
        cat_df_test = self._extract_categorical_features(df_test, features_to_extract=cats_to_use)
        
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_cat = self.one_hot_encoder.fit_transform(cat_df_train)
        X_test_cat = self.one_hot_encoder.transform(cat_df_test)
        
        feature_names_cat = list(self.one_hot_encoder.get_feature_names_out(cats_to_use))
        print(f"Created {len(feature_names_cat)} features from categorical data.")
        return X_train_cat, X_test_cat, feature_names_cat

    def _select_numerical_features(self, X_num_train, X_num_test, num_feature_names):
        """
        NEW: Selects a subset of numerical features based on `__init__` parameters.
        """
        n_features_total = X_num_train.shape[1]
        all_indices = np.arange(n_features_total)
        
        indices_to_use = None

        if self.numerical_feature_indices is not None:
            print("INFO: Using specified allowlist for numerical features.")
            invalid_indices = [i for i in self.numerical_feature_indices if not (0 <= i < n_features_total)]
            if invalid_indices:
                raise ValueError(f"numerical_feature_indices contains invalid values. Max index is {n_features_total - 1}, but got: {invalid_indices}")
            indices_to_use = sorted(list(set(self.numerical_feature_indices)))

        elif self.use_random_subset_of_numerical_features:
            print("INFO: Using random subset of numerical features.")
            n_to_select = max(1, int(n_features_total * self.subset_fraction))
            rng = np.random.default_rng(self.random_state)
            indices_to_use = sorted(rng.choice(all_indices, size=n_to_select, replace=False).tolist())
        
        else:
            print(f"INFO: Using all {n_features_total} numerical features.")
            final_names = num_feature_names
            return X_num_train, X_num_test, final_names, None # Return None for indices to signify all were used

        final_names = [num_feature_names[i] for i in indices_to_use]
        print(f"Selected {len(final_names)} numerical features out of {n_features_total}.")
        return X_num_train[:, indices_to_use], X_num_test[:, indices_to_use], final_names, indices_to_use

    def _extract_categorical_features(self, df: pd.DataFrame, features_to_extract: list[str]) -> pd.DataFrame:
        """
        REWRITTEN: Parses the JSON column and extracts only the specified categorical features.
        """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in the DataFrame.")

        categorical_data = []
        parsed_json = df.loc[:, self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        for js_obj in parsed_json:
            row_data = {key: "missing" for key in features_to_extract} # Default placeholder
            if js_obj is not None:
                for key in features_to_extract:
                    row_data[key] = js_obj.get(key, "missing")
            categorical_data.append(row_data)

        return pd.DataFrame(categorical_data, index=df.index, columns=features_to_extract)

    def _extract_test_odds(self, df_test: pd.DataFrame) -> dict:
        """Extracts betting odds from the test dataframe."""
        return {
            "team1_ml": pd.to_numeric(df_test.get("team1_moneyline"), errors='coerce').fillna(0),
            "team2_ml": pd.to_numeric(df_test.get("team2_moneyline"), errors='coerce').fillna(0),
            "team1_spread": pd.to_numeric(df_test.get("team1_spread"), errors='coerce').fillna(0),
            "team2_spread": pd.to_numeric(df_test.get("team2_spread"), errors='coerce').fillna(0),
            "team1_spread_odds": pd.to_numeric(df_test.get("team1_spread_odds"), errors='coerce').fillna(0),
            "team2_spread_odds": pd.to_numeric(df_test.get("team2_spread_odds"), errors='coerce').fillna(0),
            "total_score": pd.to_numeric(df_test.get("total_score"), errors='coerce').fillna(0),
            "over_odds": pd.to_numeric(df_test.get("over_odds"), errors='coerce').fillna(0),
            "under_odds": pd.to_numeric(df_test.get("under_odds"), errors='coerce').fillna(0)
        }

    # =================================================================================
    # Internal Helper Methods: Model & Data I/O
    # =================================================================================
    
    def _save_training_dataframe(self, X_train: np.ndarray, feature_names: list, filename: str):
        """
        Saves the final training data matrix to a CSV file for inspection.
        """
        if X_train.size == 0 or not feature_names:
            print("INFO: Training data is empty, skipping save.")
            return

        output_dir = "training_data"
        os.makedirs(output_dir, exist_ok=True)
        
        df_to_save = pd.DataFrame(X_train, columns=feature_names)

        output_path = os.path.join(output_dir, f"{filename}.csv")
        df_to_save.to_csv(output_path, index=False)
        print(f"INFO: Saved final training data to {output_path}")

    def _tune_hyperparameters(self, model, X_train, y_train):
        """
        Performs hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit.
        """
        param_grid = self._HYPERPARAMETER_GRIDS.get(self.model_type)

        if not param_grid:
            print(f"No hyperparameter grid defined for {self.model_type}. Training with default parameters.")
            model.fit(X_train, y_train)
            return model

        time_series_cv = TimeSeriesSplit(n_splits=self.tuning_cv)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=self.tuning_n_iter,
            cv=time_series_cv,
            scoring='neg_log_loss' if self.model_type in self._CLASSIFIER_TYPES else 'neg_mean_squared_error',
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1 
        )
        
        print(f"--- Starting Hyperparameter Tuning with TimeSeriesSplit (n_splits={self.tuning_cv}) ---")
        search.fit(X_train, y_train)
        
        return search.best_estimator_

    def _load_data_from_db(self, query: str) -> pd.DataFrame:
        """Connects to the database and returns a DataFrame."""
        db_path = "sports.db"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at {db_path}")
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _get_model(self):
        """Initializes a model instance from the model factory."""
        return self._MODELS[self.model_type]()

    def _save_model(self, model, scaler, query: str):
        """ REWRITTEN: Saves the model, scaler, encoder, and all feature selection metadata."""
        os.makedirs("models", exist_ok=True)
        base_path = os.path.join("models", self.model_name)
        output_path = f"{base_path}.joblib"
        
        if os.path.exists(output_path):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            archive_path = f"{base_path}_{timestamp}.joblib"
            os.rename(output_path, archive_path)
            print(f"Archived existing model to {archive_path}")

        model_info = {
            "model": model, 
            "scaler": scaler, 
            "one_hot_encoder": self.one_hot_encoder,
            "model_type": self.model_type,
            "column": self.column, 
            "query": query,
            # NEW: Save all feature selection and engineering parameters
            "feature_engineering_mode": self.feature_engineering_mode,
            "trained_numerical_indices": self.trained_numerical_indices_,
            "categorical_feature_names": self.categorical_feature_names,
            "include_market_spread": self.include_market_spread,
            "include_market_total": self.include_market_total
        }
        joblib.dump(model_info, output_path)
        print(f"Model and all metadata saved to {output_path}\n")

    def _load_model(self) -> dict:
        """Loads the most recent model file matching the model_name."""
        path = os.path.join("models", f"{self.model_name}.joblib")
        if not os.path.exists(path):
            candidates = glob.glob(os.path.join("models", f"{self.model_name}*.joblib"))
            if not candidates: raise FileNotFoundError(f"No model files found for '{self.model_name}'")
            path = max(candidates, key=os.path.getmtime)
            print(f"Loading latest versioned model from {path}")
        
        print(f"Loading model from {path}")
        return joblib.load(path)

    @staticmethod
    def _flatten_json_to_list(obj):
        """
        FIXED: Recursively flattens a JSON-like object into two lists: one for values and one for names.
        Returns a tuple of (values, names).
        """
        out_list = []
        name_list = []

        def _recursive_flatten(sub_obj, prefix):
            if isinstance(sub_obj, dict):
                for key in sorted(sub_obj.keys()):
                    _recursive_flatten(sub_obj[key], f"{prefix}.{key}" if prefix else key)
            elif isinstance(sub_obj, list):
                # For simplicity, iterate through lists. A more complex approach might aggregate.
                for i, v in enumerate(sub_obj):
                    _recursive_flatten(v, f"{prefix}.{i}")
            elif isinstance(sub_obj, bool):
                out_list.append(1.0 if sub_obj else 0.0)
                name_list.append(prefix)
            elif isinstance(sub_obj, (int, float)):
                out_list.append(float(sub_obj))
                name_list.append(prefix)
            elif sub_obj is None:
                out_list.append(0.0)
                name_list.append(prefix)
            # Note: String and other non-numeric types are skipped.

        _recursive_flatten(obj, '')
        return out_list, name_list