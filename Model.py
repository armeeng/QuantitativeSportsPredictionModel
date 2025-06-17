import os
import json
import sqlite3
import glob
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import random
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor
from TestModel import TestModel

# A simple base class for defining common attributes.
class BaseModel:
    def __init__(self, model_name: str, column: str):
        self.model_name = model_name
        self.column = column

class MLModel(BaseModel):
    """
    A simplified ML class focused strictly on training models and generating predictions.

    This class trains a model on historical sports data and makes predictions on a
    held-out test set. All metric calculations are removed. The test set predictions,
    true outcomes, and betting odds are stored as instance attributes for external evaluation.
    It supports training on a full or random subset of features.
    """
    _MODELS = {
        'linear_regression': LinearRegression,
        'random_forest': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': lambda: XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
        'mlp': lambda: MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'logistic_regression': lambda: LogisticRegression(solver='liblinear', max_iter=1000, random_state=42),
        'knn_regressor': lambda: KNeighborsRegressor(n_neighbors=5),
        'knn_classifier': lambda: KNeighborsClassifier(n_neighbors=5),
        'svr': lambda: MultiOutputRegressor(SVR(kernel='rbf')),
        'svm': lambda: SVC(probability=True, random_state=42)
    }
    _MODELS['neural_network'] = _MODELS['mlp']
    _MODELS['mlp_regressor'] = _MODELS['mlp']
    _MODELS['svc'] = _MODELS['svm']

    _CLASSIFIER_TYPES = {'logistic_regression', 'knn_classifier', 'svm', 'svc'}

    _FEATURE_KEYS_TO_DROP = {
        "team1_id", "team2_id", "venue_id", "season_type", "day", "month",
        "year", "days_since_epoch", "game_time", "day_of_week"
    }

    def __init__(self, model_name: str, model_type: str = 'linear_regression',
                 column: str = "normalized_stats", use_random_subset_of_features: bool = False,
                 subset_fraction: float = None):
        """
        Initializes the MLModel.

        Args:
            model_name (str): A unique name for saving and loading the model.
            model_type (str): The type of algorithm to use.
            column (str): The database column containing the JSON features.
            use_random_subset_of_features (bool): If True, train on a random subset of features.
            subset_fraction (float): The fraction of features to use if subsetting is enabled.
                                     If None, a random fraction is chosen.
        """
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()
        if self.model_type not in self._MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}. Supported types are {list(self._MODELS.keys())}")

        self.use_random_subset_of_features = use_random_subset_of_features
        if subset_fraction is None:
            self.subset_fraction = random.uniform(0.01, 1.0)
        else:
            if not (0 < subset_fraction <= 1.0):
                raise ValueError("If specified, subset_fraction must be > 0 and <= 1.")
            self.subset_fraction = subset_fraction
        
        # Attributes to hold test data for external evaluation
        self.predictions = None
        self.y_test = None
        self.test_odds = None
        
        # Stores the indices of features selected during training.
        self.feature_indices_ = None

    # =================================================================================
    # Public API: Main Methods
    # =================================================================================

    def train(self, query: str, test_size: float = 0.5, random_state: int = 42):
        """
        Trains a model and prepares test set predictions for external evaluation.
        Routes to the appropriate training method based on model_type.
        """
        if self.model_type in self._CLASSIFIER_TYPES:
            self._train_classifier(query, test_size, random_state)
        else:
            self._train_regressor(query, test_size, random_state)

    def predict(self, query: str) -> list:
        """
        Loads the latest model from disk and returns predictions for new, unseen data.
        This method is for inference, not for test set evaluation during training.
        """
        try:
            info = self._load_model()
            model = info['model']
            column = info['column']
            feature_indices = info.get('feature_indices')
            original_shape = info.get('original_input_shape', info.get('input_shape'))
            model_input_shape = info.get('trained_input_shape', info.get('input_shape'))
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []

        df = self._load_data_from_db(query)
        if df.empty:
            print("Query returned no data to predict on.")
            return []
        
        self.column = column
        X_raw = self._prepare_features(df)

        if X_raw.shape[1] != original_shape:
            raise ValueError(f"Feature shape mismatch: model trained on {original_shape} features, new data has {X_raw.shape[1]}.")

        X = X_raw[:, feature_indices] if feature_indices is not None else X_raw

        if X.shape[1] != model_input_shape:
            raise ValueError(f"Model input shape mismatch: model expects {model_input_shape} features, got {X.shape[1]}.")

        if isinstance(model, dict):  # Classifier
            prob_win = model['win'].predict_proba(X)[:, 1]
            prob_cover = model['spread'].predict_proba(X)[:, 1]
            prob_over = model['over'].predict_proba(X)[:, 1]
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "team1_win_prob": pw, "team1_cover_prob": pc, "over_prob": po}
                for gid, t1, t2, pw, pc, po in zip(df["game_id"], df["team1_id"], df["team2_id"], prob_win, prob_cover, prob_over)
            ]
        else:  # Regressor
            predictions = model.predict(X)
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "pred_team1": p1, "pred_team2": p2}
                for gid, t1, t2, (p1, p2) in zip(df["game_id"], df["team1_id"], df["team2_id"], predictions)
            ]

    # =================================================================================
    # Internal Training Methods
    # =================================================================================

    def _train_regressor(self, query: str, test_size: float, random_state: int):
        """Trains a regression model and stores test set results."""
        df = self._load_data_from_db(query)
        df.dropna(subset=["team1_score", "team2_score"], inplace=True)
        if df.empty:
            print("Query returned no data to train on. Aborting.")
            return

        X_raw = self._prepare_features(df)
        original_feature_count = X_raw.shape[1]
        X = self._select_features(X_raw, random_state)
        y = df[["team1_score", "team2_score"]].to_numpy()

        betting_cols = {
            "team1_ml": df.get("team1_moneyline"), "team2_ml": df.get("team2_moneyline"),
            "team1_spread": df.get("team1_spread"), "team2_spread": df.get("team2_spread"),
            "team1_spread_odds": df.get("team1_spread_odds"), "team2_spread_odds": df.get("team2_spread_odds"),
            "total_score": df.get("total_score"), "over_odds": df.get("over_odds"), "under_odds": df.get("under_odds")
        }
        
        data_to_split = [X, y] + [pd.to_numeric(col, errors='coerce').fillna(0) for col in betting_cols.values()]
        splits = train_test_split(*data_to_split, test_size=test_size, random_state=random_state)
        
        X_train, X_test, y_train, self.y_test = splits[0], splits[1], splits[2], splits[3]
        
        # Store test odds for external evaluation
        self.test_odds = dict(zip(betting_cols.keys(), splits[5::2]))

        model = self._get_model()
        model.fit(X_train, y_train)

        # Generate and store predictions on the test set
        self.predictions = model.predict(X_test)

        test_evaluator = TestModel(
            predictions=self.predictions,
            y_test=self.y_test,
            test_odds=self.test_odds
        )

        # 5. Display the results
        test_evaluator.display_results()

        self._save_model(model, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) on {len(X_train)} train / {len(X_test)} test games.")
        print("Test predictions and data are now available in instance attributes (e.g., model.predictions).")


    def _train_classifier(self, query: str, test_size: float, random_state: int):
        """Trains classification models and stores test set results."""
        df = self._load_data_from_db(query)
        df.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)
        if df.empty:
            print("Query returned no data to train on. Aborting.")
            return

        X_raw = self._prepare_features(df)
        original_feature_count = X_raw.shape[1]
        X = self._select_features(X_raw, random_state)

        actual_margin = df["team1_score"] - df["team2_score"]
        actual_total = df["team1_score"] + df["team2_score"]
        y_win = (actual_margin > 0).astype(int)
        y_spread_outcome = actual_margin + pd.to_numeric(df["team1_spread"], errors='coerce').fillna(0)
        y_total_outcome = actual_total - pd.to_numeric(df["total_score"], errors='coerce').fillna(0)

        betting_cols = {
            "team1_ml": df.get("team1_moneyline"), "team2_ml": df.get("team2_moneyline"),
            "team1_spread": df.get("team1_spread"), "team2_spread": df.get("team2_spread"),
            "team1_spread_odds": df.get("team1_spread_odds"), "team2_spread_odds": df.get("team2_spread_odds"),
            "total_score": df.get("total_score"), "over_odds": df.get("over_odds"), "under_odds": df.get("under_odds")
        }
        
        data_to_split = [X, y_win, y_spread_outcome, y_total_outcome, df[["team1_score", "team2_score"]]] + \
                        [pd.to_numeric(col, errors='coerce').fillna(0) for col in betting_cols.values()]
        
        splits = train_test_split(*data_to_split, test_size=test_size, random_state=random_state, stratify=y_win)
        X_train, X_test = splits[0], splits[1]
        y_train_win, _ = splits[2], splits[3] # y_test_win is not needed directly
        y_train_spread_outcome, _ = splits[4], splits[5]
        y_train_total_outcome, _ = splits[6], splits[7]
        
        # Store the actual scores for the test set
        self.y_test = splits[9] 
        # Store test odds for external evaluation
        self.test_odds = dict(zip(betting_cols.keys(), splits[11::2]))

        # Train three separate models
        model_win = self._get_model().fit(X_train, y_train_win)
        
        # Filter out pushes for spread and total training data
        train_spread_non_push = y_train_spread_outcome != 0
        model_spread = self._get_model().fit(X_train[train_spread_non_push], (y_train_spread_outcome[train_spread_non_push] > 0).astype(int))

        train_total_non_push = y_train_total_outcome != 0
        model_over = self._get_model().fit(X_train[train_total_non_push], (y_train_total_outcome[train_total_non_push] > 0).astype(int))

        models = {'win': model_win, 'spread': model_spread, 'over': model_over}
        
        # Generate and store predictions (probabilities) on the test set
        self.predictions = {
            'win': models['win'].predict_proba(X_test),
            'spread': models['spread'].predict_proba(X_test),
            'over': models['over'].predict_proba(X_test)
        }

        test_evaluator = TestModel(
            predictions=self.predictions,
            y_test=self.y_test,
            test_odds=self.test_odds
        )

        # 5. Display the results
        test_evaluator.display_results()

        self._save_model(models, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) with {len(X_train)} train / {len(X_test)} test games.")
        print("Test predictions and data are now available in instance attributes (e.g., model.predictions).")


    # =================================================================================
    # Internal Helper Methods
    # =================================================================================

    def _select_features(self, X: np.ndarray, random_state: int) -> np.ndarray:
        """Selects a random subset of features if enabled."""
        if self.use_random_subset_of_features:
            n_features = X.shape[1]
            n_selected_features = max(1, int(n_features * self.subset_fraction))

            rng = np.random.default_rng()
            indices = rng.choice(n_features, size=n_selected_features, replace=False)
            indices.sort()
            
            self.feature_indices_ = indices
            print(f"INFO: Using a random subset of {len(self.feature_indices_)} out of {n_features} features.")
            return X[:, self.feature_indices_]
        else:
            self.feature_indices_ = None
            return X

    def _load_data_from_db(self, query: str) -> pd.DataFrame:
        """Connects to the database and returns a DataFrame."""
        # Ensure the database file exists or handle creation
        db_path = "sports.db"
        if not os.path.exists(db_path):
             raise FileNotFoundError(f"Database file not found at {db_path}")
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Parses and flattens a JSON column into a feature matrix.
        Raises an error if the resulting feature vectors have inconsistent lengths.
        """
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in the DataFrame.")

        parsed_json = df[self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        feature_list = []
        for idx, js_obj in enumerate(parsed_json):
            if js_obj is None:
                # We add a print statement to help debug which row is causing issues
                print(f"Warning: JSON object is None for DataFrame index {df.index[idx]}. Skipping.")
                continue
            temp_obj = js_obj.copy()
            for key in self._FEATURE_KEYS_TO_DROP:
                temp_obj.pop(key, None)

            row_features = []
            self._flatten_json_to_list(temp_obj, row_features)
            feature_list.append(row_features)

        if not feature_list:
            return np.array([])

        # --- REPLACEMENT BLOCK ---
        # Validate that all feature vectors have the same length.
        it = iter(feature_list)
        first_row_len = len(next(it, []))
        if not all(len(row) == first_row_len for row in it):
            # Find and show the lengths to help debug the source data
            all_lengths = {len(r) for r in feature_list}
            raise ValueError(
                f"Inconsistent feature lengths detected. All rows must produce a feature vector of the same size. "
                f"Found lengths: {all_lengths}. This suggests the JSON data structure is not consistent across all rows. "
                f"Please fix the upstream data generation process."
            )
        # --- END REPLACEMENT BLOCK ---

        return np.array(feature_list, dtype=float)

    def _get_model(self):
        """Initializes a model instance from the model factory."""
        return self._MODELS[self.model_type]()

    def _save_model(self, model, query: str, trained_input_shape: int, original_input_shape: int):
        """Saves the model and metadata to a versioned .joblib file."""
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
            "model_type": self.model_type,
            "trained_input_shape": trained_input_shape,
            "original_input_shape": original_input_shape,
            "feature_indices": self.feature_indices_,
            "column": self.column,
            "query": query,
        }
        joblib.dump(model_info, output_path)
        print(f"Model saved to {output_path}\n")

    def _load_model(self) -> dict:
        """Loads the most recent model file matching the model_name."""
        path = os.path.join("models", f"{self.model_name}.joblib")
        if not os.path.exists(path):
            pattern = os.path.join("models", f"{self.model_name}*.joblib")
            candidates = glob.glob(pattern)
            if not candidates:
                raise FileNotFoundError(f"No model files found for '{self.model_name}'")
            latest_file = max(candidates, key=os.path.getmtime)
            print(f"Loading latest versioned model from {latest_file}")
            return joblib.load(latest_file)
        
        print(f"Loading model from {path}")
        return joblib.load(path)

    @staticmethod
    def _flatten_json_to_list(obj, out_list: list):
        """Recursively flattens a JSON object (dict/list) into a single list."""
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                MLModel._flatten_json_to_list(obj[key], out_list)
        elif isinstance(obj, list):
            for v in obj:
                MLModel._flatten_json_to_list(v, out_list)
        elif isinstance(obj, bool):
            out_list.append(1.0 if obj else 0.0)
        elif isinstance(obj, (int, float)):
            out_list.append(float(obj))
        elif obj is None:
            out_list.append(0.0)
