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
from sklearn.preprocessing import StandardScaler
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
    _MODELS = {
        # Regressors
        'linear_regression': LinearRegression,
        'random_forest_regressor': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost_regressor': lambda: XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        'mlp_regressor': lambda: MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'knn_regressor': lambda: KNeighborsRegressor(n_neighbors=5),
        'svr': lambda: MultiOutputRegressor(SVR(kernel='rbf')),

        # Classifiers
        'logistic_regression': lambda: LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
        'knn_classifier': lambda: KNeighborsClassifier(n_neighbors=5),
        'svc': lambda: SVC(probability=True, random_state=42),
        'random_forest_classifier': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
        'xgboost_classifier': lambda: XGBClassifier(objective='binary:logistic', n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42),
        'mlp_classifier': lambda: MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'gradient_boosting_classifier': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        'gaussian_nb': lambda: GaussianNB(),
    }
    # Legacy aliases
    _MODELS['random_forest'] = _MODELS['random_forest_regressor']
    _MODELS['xgboost'] = _MODELS['xgboost_regressor']
    _MODELS['mlp'] = _MODELS['mlp_regressor']
    _MODELS['neural_network'] = _MODELS['mlp_regressor']
    _MODELS['svm'] = _MODELS['svc']

    # =================================================================================
    # NEW: HYPERPARAMETER GRIDS FOR RandomizedSearchCV
    # =================================================================================
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
    # Add aliases for hyperparameter grids
    _HYPERPARAMETER_GRIDS['random_forest'] = _HYPERPARAMETER_GRIDS['random_forest_regressor']
    _HYPERPARAMETER_GRIDS['xgboost'] = _HYPERPARAMETER_GRIDS['xgboost_regressor']
    _HYPERPARAMETER_GRIDS['mlp'] = _HYPERPARAMETER_GRIDS['mlp_regressor']
    _HYPERPARAMETER_GRIDS['neural_network'] = _HYPERPARAMETER_GRIDS['mlp_regressor']
    _HYPERPARAMETER_GRIDS['svm'] = _HYPERPARAMETER_GRIDS['svc']

    _CLASSIFIER_TYPES = {
        'logistic_regression', 'knn_classifier', 'svc', 'random_forest_classifier',
        'xgboost_classifier', 'mlp_classifier', 'gradient_boosting_classifier', 'gaussian_nb'
    }

    _FEATURE_KEYS_TO_DROP = {
        "team1_id", "team2_id", "venue_id", "season_type", "day", "month",
        "year", "days_since_epoch", "game_time", "day_of_week"
    }

    def __init__(self, model_name: str, model_type: str = 'linear_regression',
                 column: str = "normalized_stats", use_random_subset_of_features: bool = False,
                 subset_fraction: float = None, feature_allowlist: list[int] = None,
                 hyperparameter_tuning: bool = False, tuning_n_iter: int = 50, tuning_cv: int = 5):
        """
        MODIFIED: `__init__` now accepts hyperparameter tuning options.
        
        Args:
            hyperparameter_tuning (bool): If True, run RandomizedSearchCV to find the best hyperparameters.
            tuning_n_iter (int): The number of parameter settings that are sampled. `n_iter` trades off runtime vs quality of the solution.
            tuning_cv (int): The number of folds to use for cross-validation during tuning.
        """
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()
        if self.model_type not in self._MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}. Supported types are {list(self._MODELS.keys())}")

        if use_random_subset_of_features and feature_allowlist is not None:
            raise ValueError("Cannot set `use_random_subset_of_features` to True and provide a `feature_allowlist` simultaneously.")

        self.use_random_subset_of_features = use_random_subset_of_features
        self.feature_allowlist = feature_allowlist

        if subset_fraction is None: self.subset_fraction = random.uniform(0.01, 1.0)
        else:
            if not (0 < subset_fraction <= 1.0): raise ValueError("If specified, subset_fraction must be > 0 and <= 1.")
            self.subset_fraction = subset_fraction

        # NEW: Store tuning parameters
        self.hyperparameter_tuning = hyperparameter_tuning
        self.tuning_n_iter = tuning_n_iter
        self.tuning_cv = tuning_cv

        self.predictions = None
        self.y_test = None
        self.test_odds = None
        self.feature_indices_ = None
        self.scaler = None
        self.feature_names_ = None
        self.model_ = None

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
        Loads the latest model and scaler from disk and returns predictions for new, unseen data.
        This method is for inference, not for test set evaluation during training.
        """
        try:
            info = self._load_model()
            model = info['model']
            self.scaler = info.get('scaler')
            column = info['column']
            feature_indices = info.get('feature_indices')
            original_shape = info.get('original_input_shape')
            model_input_shape = info.get('trained_input_shape')
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []

        df = self._load_data_from_db(query)
        if df.empty:
            print("Query returned no data to predict on.")
            return []
        
        self.column = column
        X_raw, _ = self._prepare_features(df)

        if X_raw.shape[1] != original_shape:
            raise ValueError(f"Feature shape mismatch: model trained on {original_shape} features, new data has {X_raw.shape[1]}.")

        X = X_raw[:, feature_indices] if feature_indices is not None else X_raw

        if X.shape[1] != model_input_shape:
            raise ValueError(f"Model input shape mismatch: model expects {model_input_shape} features, got {X.shape[1]}.")

        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            print("Warning: No scaler found. Predicting on unscaled data.")
            X_scaled = X

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
        MODIFIED: Calculates and returns feature importances, including the feature index
        from the original flattened input array.
        """
        if model is None: model = self.model_
        if self.feature_names_ is None:
            print("Feature names are not available. Please train the model first.")
            return None

        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if model.coef_.ndim > 1: importances = np.mean(np.abs(model.coef_), axis=0)
            else: importances = np.abs(model.coef_)
        elif X_test is not None and y_test is not None:
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances = result.importances_mean
        else:
            print(f"Cannot get feature importance for model type {type(model).__name__} without test data.")
            return None

        # Determine the original indices of the features used by the model.
        if self.feature_indices_ is not None:
            # A subset of features was selected during training.
            feature_indices = self.feature_indices_
        else:
            # All features were used; indices are a simple range.
            feature_indices = list(range(len(self.feature_names_)))

        return pd.DataFrame({
            'feature_index': feature_indices,
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

    # =================================================================================
    # Internal Training Methods
    # =================================================================================

    def _train_regressor(self, train_query: str, test_query: str):
        """Trains a regression model and evaluates on a separate test set."""
        # Load Data
        df_train = self._load_data_from_db(train_query)
        df_test = self._load_data_from_db(test_query)

        df_train.dropna(subset=["team1_score", "team2_score"], inplace=True)
        df_test.dropna(subset=["team1_score", "team2_score"], inplace=True)
        
        if df_train.empty:
            print("Training query returned no data. Aborting.")
            return
        if df_test.empty:
            print("Test query returned no data. Aborting.")
            return

        # Prepare Training Data
        X_train_raw, self.feature_names_ = self._prepare_features(df_train)
        original_feature_count = X_train_raw.shape[1]
        X_train = self._select_features(X_train_raw, random_state=42) # Use fixed random state for reproducibility
        y_train = df_train[["team1_score", "team2_score"]].to_numpy()

        # Prepare Test Data
        X_test_raw, _ = self._prepare_features(df_test)
        if X_test_raw.shape[1] != original_feature_count:
            raise ValueError(f"Feature count mismatch: train data has {original_feature_count} features, test data has {X_test_raw.shape[1]}.")
        
        X_test = X_test_raw[:, self.feature_indices_] if self.feature_indices_ is not None else X_test_raw
        self.y_test = df_test[["team1_score", "team2_score"]].to_numpy()
        
        self.test_odds = {
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

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Model
        # --- MODIFIED: Model Training Step ---
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
        # ----------------------------------------

        # Evaluate Model
        self.predictions = self.model_.predict(X_test_scaled)
        test_evaluator = TestModel(predictions=self.predictions, y_test=self.y_test, test_odds=self.test_odds)
        test_evaluator.display_results()
        
        # Feature Importance
        feature_importance = self.get_feature_importance(model=self.model_, X_test=X_test_scaled, y_test=self.y_test)
        if feature_importance is not None:
            feature_importance.to_csv("Feature Importance/feature_importance.csv", index=False)
            print("Saved feature importance to feature_importance.csv")

        self._save_model(self.model_, self.scaler, train_query, X_train.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) on {len(X_train)} train / {len(X_test)} test games.")

    def _train_classifier(self, train_query: str, test_query: str):
        """Trains classification models and evaluates on a separate test set."""
        # Load Data
        df_train = self._load_data_from_db(train_query)
        df_test = self._load_data_from_db(test_query)

        df_train.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)
        df_test.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)

        if df_train.empty:
            print("Training query returned no data. Aborting.")
            return
        if df_test.empty:
            print("Test query returned no data. Aborting.")
            return

        # Prepare Training Data from JSON
        X_train_raw, self.feature_names_ = self._prepare_features(df_train)
        
        # Add spread and total features to the matrix
        spread_feature_train = pd.to_numeric(df_train["team1_spread"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
        total_feature_train = pd.to_numeric(df_train["total_score"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
        X_train_raw = np.hstack((X_train_raw, spread_feature_train, total_feature_train))
        self.feature_names_.extend(['market_team1_spread', 'market_total_score'])
        
        # <<< START: MODIFICATION TO FORCE INCLUDE SPREAD/TOTAL >>>
        # Get the indices of the features we just added. They will be the last two.
        original_feature_count = X_train_raw.shape[1]
        forced_indices = [original_feature_count - 2, original_feature_count - 1]

        # Call _select_features, passing the indices to force include
        X_train = self._select_features(X_train_raw, random_state=42, force_include_indices=forced_indices)
        # <<< END: MODIFICATION >>>

        # --- The rest of the function proceeds as before ---

        y_train_win = (df_train["team1_score"] > df_train["team2_score"]).astype(int)
        # ... (y_train_spread_outcome, y_train_total_outcome definitions) ...
        y_train_spread_outcome = (df_train["team1_score"] - df_train["team2_score"]) + pd.to_numeric(df_train["team1_spread"], errors='coerce').fillna(0)
        y_train_total_outcome = (df_train["team1_score"] + df_train["team2_score"]) - pd.to_numeric(df_train["total_score"], errors='coerce').fillna(0)

        # Prepare Test Data
        X_test_raw, _ = self._prepare_features(df_test)
        spread_feature_test = pd.to_numeric(df_test["team1_spread"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
        total_feature_test = pd.to_numeric(df_test["total_score"], errors='coerce').fillna(0).to_numpy().reshape(-1, 1)
        X_test_raw = np.hstack((X_test_raw, spread_feature_test, total_feature_test))

        if X_test_raw.shape[1] != original_feature_count:
            raise ValueError(f"Feature count mismatch: train data has {original_feature_count} features, test data has {X_test_raw.shape[1]}.")
        
        X_test = X_test_raw[:, self.feature_indices_] if self.feature_indices_ is not None else X_test_raw
        
        self.y_test = df_test[["team1_score", "team2_score"]]
        y_test_win = (df_test["team1_score"] > df_test["team2_score"]).astype(int)
        y_test_spread_outcome = (df_test["team1_score"] - df_test["team2_score"]) + pd.to_numeric(df_test["team1_spread"], errors='coerce').fillna(0)
        y_test_total_outcome = (df_test["team1_score"] + df_test["team2_score"]) - pd.to_numeric(df_test["total_score"], errors='coerce').fillna(0)
        
        self.test_odds = {
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

        # Scale Features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Models
        # --- MODIFIED: Model Training Step ---
        # Train Win Model
        initial_win_model = self._get_model()
        if self.hyperparameter_tuning:
            print(f"\n--- Tuning WIN model ({self.model_type}) ---")
            model_win = self._tune_hyperparameters(initial_win_model, X_train_scaled, y_train_win)
            print(f"Best WIN model parameters: {model_win.get_params()}")
        else:
            model_win = initial_win_model.fit(X_train_scaled, y_train_win)

        # Train Spread Model
        train_spread_non_push = y_train_spread_outcome != 0
        initial_spread_model = self._get_model()
        if self.hyperparameter_tuning:
            print(f"\n--- Tuning SPREAD model ({self.model_type}) ---")
            model_spread = self._tune_hyperparameters(
                initial_spread_model,
                X_train_scaled[train_spread_non_push],
                (y_train_spread_outcome[train_spread_non_push] > 0).astype(int)
            )
            print(f"Best SPREAD model parameters: {model_spread.get_params()}")
        else:
            model_spread = initial_spread_model.fit(
                X_train_scaled[train_spread_non_push], 
                (y_train_spread_outcome[train_spread_non_push] > 0).astype(int)
            )

        # Train Over/Under Model
        train_total_non_push = y_train_total_outcome != 0
        initial_over_model = self._get_model()
        if self.hyperparameter_tuning:
            print(f"\n--- Tuning OVER/UNDER model ({self.model_type}) ---")
            model_over = self._tune_hyperparameters(
                initial_over_model,
                X_train_scaled[train_total_non_push],
                (y_train_total_outcome[train_total_non_push] > 0).astype(int)
            )
            print(f"Best OVER/UNDER model parameters: {model_over.get_params()}")
        else:
            model_over = initial_over_model.fit(
                X_train_scaled[train_total_non_push],
                (y_train_total_outcome[train_total_non_push] > 0).astype(int)
            )
        
        print("\n--- Hyperparameter Tuning Finished for all models ---\n")
        self.model_ = {'win': model_win, 'spread': model_spread, 'over': model_over}
        # ----------------------------------------
        
        # --- Evaluation and Saving (Unchanged) ---
        self.predictions = {
            'win': self.model_['win'].predict_proba(X_test_scaled),
            'spread': self.model_['spread'].predict_proba(X_test_scaled),
            'over': self.model_['over'].predict_proba(X_test_scaled)
        }
        test_evaluator = TestModel(predictions=self.predictions, y_test=self.y_test, test_odds=self.test_odds)
        test_evaluator.display_results()
        
        # Feature Importance
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

        self._save_model(self.model_, self.scaler, train_query, X_train.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) with {len(X_train)} train / {len(X_test)} test games.")


    # =================================================================================
    # Internal Helper Methods
    # =================================================================================

    # NEW: Method to handle hyperparameter tuning
    def _tune_hyperparameters(self, model, X_train, y_train):
        """
        Performs hyperparameter tuning using RandomizedSearchCV with TimeSeriesSplit
        to respect the temporal order of the data.

        Args:
            model: The scikit-learn model instance to tune.
            X_train: The training feature data, MUST be sorted chronologically.
            y_train: The training target data.

        Returns:
            The best model found by the search, refit on the entire training data.
        """
        param_grid = self._HYPERPARAMETER_GRIDS.get(self.model_type)

        if not param_grid:
            print(f"No hyperparameter grid defined for {self.model_type}. Training with default parameters.")
            model.fit(X_train, y_train)
            return model

        # =========================================================================
        # KEY CHANGE: Use TimeSeriesSplit instead of a simple integer for 'cv'
        # =========================================================================
        time_series_cv = TimeSeriesSplit(n_splits=self.tuning_cv)

        # For MultiOutputRegressor (like SVR), y_train can be 2D. RandomizedSearchCV handles this.
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=self.tuning_n_iter,
            # Pass the TimeSeriesSplit object here
            cv=time_series_cv,
            scoring='neg_log_loss' if self.model_type in self._CLASSIFIER_TYPES else 'neg_mean_squared_error',
            verbose=1, # Set to 2 for more details
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )
        
        print(f"--- Starting Hyperparameter Tuning with TimeSeriesSplit (n_splits={self.tuning_cv}) ---")
        search.fit(X_train, y_train)
        
        # The search object automatically refits the best model on the whole dataset
        return search.best_estimator_
    
    def _select_features(self, X: np.ndarray, random_state: int, force_include_indices: list[int] = None) -> np.ndarray:
        """
        MODIFIED: Selects features based on initialization settings, now with an option
        to force the inclusion of specific feature indices.

        Priority:
        1. feature_allowlist: Uses the union of the allowlist and forced indices.
        2. random_subset: Uses the union of a random subset and forced indices.
        3. all features: Uses all features (forced indices are naturally included).
        """
        n_features_total = X.shape[1]
        if force_include_indices is None:
            force_include_indices = []

        # Validate that forced indices are within the valid range.
        invalid_forced_indices = [i for i in force_include_indices if not (0 <= i < n_features_total)]
        if invalid_forced_indices:
            raise ValueError(f"Forced indices contains invalid values. Max index is {n_features_total - 1}, but got: {invalid_forced_indices}")

        indices_to_use = []

        if self.feature_allowlist:
            print(f"INFO: Using feature allowlist, ensuring forced indices are included.")
            
            # Validate that all provided allowlist indices are within the valid range.
            invalid_indices = [i for i in self.feature_allowlist if not (0 <= i < n_features_total)]
            if invalid_indices:
                raise ValueError(f"Feature allowlist contains invalid indices. Max index is {n_features_total - 1}, but got: {invalid_indices}")

            # Combine the user's allowlist with the forced indices
            final_indices_set = set(self.feature_allowlist) | set(force_include_indices)
            indices_to_use = sorted(list(final_indices_set))

        elif self.use_random_subset_of_features:
            print(f"INFO: Using random feature subset, ensuring forced indices are included.")
            
            # We must select enough random features to account for any overlap with forced indices.
            n_forced = len(force_include_indices)
            n_to_select_randomly = max(1, int(n_features_total * self.subset_fraction))

            # Exclude forced indices from the pool of potential random choices
            available_indices = np.setdiff1d(np.arange(n_features_total), force_include_indices)
            
            # Calculate how many more we need to pick
            n_random_needed = max(0, n_to_select_randomly - n_forced)

            rng = np.random.default_rng(random_state)
            if n_random_needed > 0 and len(available_indices) > 0:
                 # Ensure we don't try to pick more than are available
                n_random_to_pick = min(n_random_needed, len(available_indices))
                random_indices = rng.choice(available_indices, size=n_random_to_pick, replace=False)
                final_indices_set = set(random_indices) | set(force_include_indices)
            else:
                # Use only the forced indices if no more are needed or available
                final_indices_set = set(force_include_indices)

            indices_to_use = sorted(list(final_indices_set))

        else:
            # Use all features. The forced indices are already included by default.
            self.feature_indices_ = None
            print(f"INFO: Using all {n_features_total} features.")
            return X

        # Set the final indices and names for the selected features
        self.feature_indices_ = indices_to_use
        self.feature_names_ = [self.feature_names_[i] for i in indices_to_use]
        
        print(f"INFO: Final feature count is {len(indices_to_use)}.")
        return X[:, indices_to_use]

    def _load_data_from_db(self, query: str) -> pd.DataFrame:
        """Connects to the database and returns a DataFrame."""
        db_path = "sports.db"
        if not os.path.exists(db_path):
                 raise FileNotFoundError(f"Database file not found at {db_path}")
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list]:
        """
        Parses and flattens a JSON column into a feature matrix and a list of feature names.
        """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in the DataFrame.")

        parsed_json = df[self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        feature_list = []
        feature_names = None

        for idx, js_obj in enumerate(parsed_json):
            if js_obj is None:
                print(f"Warning: JSON object is None for DataFrame index {df.index[idx]}. Skipping.")
                continue
            
            temp_obj = js_obj.copy()
            for key in self._FEATURE_KEYS_TO_DROP: temp_obj.pop(key, None)

            row_features, row_feature_names = [], []
            
            if feature_names is None:
                self._flatten_json_to_list(temp_obj, row_features, row_feature_names, generate_names=True)
                feature_names = row_feature_names
            else:
                self._flatten_json_to_list(temp_obj, row_features)
            feature_list.append(row_features)

        if not feature_list: return np.array([]), []

        first_row_len = len(feature_list[0])
        if not all(len(row) == first_row_len for row in feature_list):
            raise ValueError("Inconsistent feature lengths detected across rows.")
        
        return np.array(feature_list, dtype=float), feature_names

    def _get_model(self):
        """Initializes a model instance from the model factory."""
        return self._MODELS[self.model_type]()

    def _save_model(self, model, scaler, query: str, trained_input_shape: int, original_input_shape: int):
        """Saves the model, scaler, and metadata to a versioned .joblib file."""
        os.makedirs("models", exist_ok=True)
        base_path = os.path.join("models", self.model_name)
        output_path = f"{base_path}.joblib"
        
        if os.path.exists(output_path):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            archive_path = f"{base_path}_{timestamp}.joblib"
            os.rename(output_path, archive_path)
            print(f"Archived existing model to {archive_path}")

        model_info = {
            "model": model, "scaler": scaler, "model_type": self.model_type,
            "trained_input_shape": trained_input_shape,
            "original_input_shape": original_input_shape,
            "feature_indices": self.feature_indices_, "column": self.column, "query": query,
        }
        joblib.dump(model_info, output_path)
        print(f"Model and scaler saved to {output_path}\n")

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
    def _flatten_json_to_list(obj, out_list: list, name_list: list = None, prefix: str = '', generate_names: bool = False):
        """Recursively flattens a JSON object/list into a single list."""
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                MLModel._flatten_json_to_list(obj[key], out_list, name_list, f"{prefix}.{key}" if prefix else key, generate_names)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                MLModel._flatten_json_to_list(v, out_list, name_list, f"{prefix}.{i}", generate_names)
        elif isinstance(obj, bool):
            out_list.append(1.0 if obj else 0.0)
            if generate_names: name_list.append(prefix)
        elif isinstance(obj, (int, float)):
            out_list.append(float(obj))
            if generate_names: name_list.append(prefix)
        elif obj is None:
            out_list.append(0.0)
            if generate_names: name_list.append(prefix)