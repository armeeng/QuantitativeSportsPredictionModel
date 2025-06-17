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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor
from sklearn.inspection import permutation_importance # Import permutation_importance
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

    This class trains a model on historical sports data and makes predictions on a
    held-out test set. All metric calculations are removed. The test set predictions,
    true outcomes, and betting odds are stored as instance attributes for external evaluation.
    It supports training on a full or random subset of features and uses StandardScaler
    to scale features.
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
                 subset_fraction: float = None, feature_allowlist: list[str] = None):
        """
        Initializes the MLModel.

        Args:
            model_name (str): A unique name for saving and loading the model.
            model_type (str): The type of algorithm to use.
            column (str): The database column containing the JSON features.
            use_random_subset_of_features (bool): If True, train on a random subset of features.
            subset_fraction (float): The fraction of features to use if subsetting is enabled.
            feature_allowlist (list[str], optional): A specific list of feature names to use for training.
        """
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()
        if self.model_type not in self._MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}. Supported types are {list(self._MODELS.keys())}")

        # NEW: Add validation for feature selection methods
        if use_random_subset_of_features and feature_allowlist is not None:
            raise ValueError("Cannot set `use_random_subset_of_features` to True and provide a `feature_allowlist` simultaneously.")

        self.use_random_subset_of_features = use_random_subset_of_features
        self.feature_allowlist = feature_allowlist  # NEW: Store the allowlist

        if subset_fraction is None:
            self.subset_fraction = random.uniform(0.01, 1.0)
        else:
            if not (0 < subset_fraction <= 1.0):
                raise ValueError("If specified, subset_fraction must be > 0 and <= 1.")
            self.subset_fraction = subset_fraction

        # ... (rest of the __init__ method is the same) ...
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

    def train(self, query: str, test_size: float = 0.2, random_state: int = 42):
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
        Loads the latest model and scaler from disk and returns predictions for new, unseen data.
        This method is for inference, not for test set evaluation during training.
        """
        try:
            info = self._load_model()
            model = info['model']
            self.scaler = info.get('scaler') # Use .get() for backward compatibility
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
        X_raw, _ = self._prepare_features(df) # We don't need feature names here

        if X_raw.shape[1] != original_shape:
            raise ValueError(f"Feature shape mismatch: model trained on {original_shape} features, new data has {X_raw.shape[1]}.")

        X = X_raw[:, feature_indices] if feature_indices is not None else X_raw

        if X.shape[1] != model_input_shape:
            raise ValueError(f"Model input shape mismatch: model expects {model_input_shape} features, got {X.shape[1]}.")

        # Scale the features using the loaded scaler
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            # If no scaler was saved, use the unscaled data (legacy model)
            print("Warning: No scaler found in the model file. Predicting on unscaled data.")
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
            
    # New method for feature importance
    def get_feature_importance(self, model=None, X_test=None, y_test=None, n_top_features=20):
        """
        Calculates and returns feature importances for the trained model.

        Args:
            model: The trained model. If None, uses self.model_.
            X_test: The test features. If None, this will be skipped for permutation importance.
            y_test: The test labels. If None, this will be skipped for permutation importance.
            n_top_features (int): The number of top features to display.

        Returns:
            A pandas DataFrame with feature names and their importance scores.
        """
        if model is None:
            model = self.model_

        if self.feature_names_ is None:
            print("Feature names are not available. Please train the model first.")
            return None

        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, we use the absolute value of the coefficients
            if model.coef_.ndim > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
        elif X_test is not None and y_test is not None:
            # Use permutation importance for models without direct importance attributes
            result = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
            )
            importances = result.importances_mean
        else:
            print(f"Cannot get feature importance for model type {type(model).__name__} without test data for permutation.")
            return None

        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        return feature_importance_df#.head(n_top_features)

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

        X_raw, self.feature_names_ = self._prepare_features(df)
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
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test odds for external evaluation
        self.test_odds = dict(zip(betting_cols.keys(), splits[5::2]))

        self.model_ = self._get_model()
        self.model_.fit(X_train_scaled, y_train)

        # Generate and store predictions on the scaled test set
        self.predictions = self.model_.predict(X_test_scaled)

        test_evaluator = TestModel(
            predictions=self.predictions,
            y_test=self.y_test,
            test_odds=self.test_odds
        )
        test_evaluator.display_results()
        
        feature_importance = self.get_feature_importance(
            model=self.model_,
            X_test=X_test_scaled,
            y_test=self.y_test
        )

        # Save to CSV
        feature_importance.to_csv("Feature Importance/feature_importance.csv", index=False)
        print("Saved feature importance to feature_importance.csv")

        self._save_model(self.model_, self.scaler, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) on {len(X_train)} train / {len(X_test)} test games.")
        print("Test predictions and data are now available in instance attributes (e.g., model.predictions).")


    def _train_classifier(self, query: str, test_size: float, random_state: int):
        """Trains classification models and stores test set results."""
        df = self._load_data_from_db(query)
        df.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)
        if df.empty:
            print("Query returned no data to train on. Aborting.")
            return

        X_raw, self.feature_names_ = self._prepare_features(df)
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
        y_train_win, y_test_win = splits[2], splits[3]
        y_train_spread_outcome, y_test_spread_outcome = splits[4], splits[5]
        y_train_total_outcome, y_test_total_outcome = splits[6], splits[7]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Store the actual scores for the test set
        self.y_test = splits[9] 
        # Store test odds for external evaluation
        self.test_odds = dict(zip(betting_cols.keys(), splits[11::2]))

        # Train three separate models on scaled data
        model_win = self._get_model().fit(X_train_scaled, y_train_win)
        
        # Filter out pushes for spread and total training data
        train_spread_non_push = y_train_spread_outcome != 0
        model_spread = self._get_model().fit(X_train_scaled[train_spread_non_push], (y_train_spread_outcome[train_spread_non_push] > 0).astype(int))

        train_total_non_push = y_train_total_outcome != 0
        model_over = self._get_model().fit(X_train_scaled[train_total_non_push], (y_train_total_outcome[train_total_non_push] > 0).astype(int))

        self.model_ = {'win': model_win, 'spread': model_spread, 'over': model_over}
        
        # Generate and store predictions (probabilities) on the scaled test set
        self.predictions = {
            'win': self.model_['win'].predict_proba(X_test_scaled),
            'spread': self.model_['spread'].predict_proba(X_test_scaled),
            'over': self.model_['over'].predict_proba(X_test_scaled)
        }

        test_evaluator = TestModel(
            predictions=self.predictions,
            y_test=self.y_test,
            test_odds=self.test_odds
        )
        test_evaluator.display_results()
        
        df_win = self.get_feature_importance(
            model=model_win,
            X_test=X_test_scaled,
            y_test=y_test_win
        )
        df_win.to_csv("Feature Importance/feature_importance_win.csv", index=False)
        print("\nSaved Win model importances to feature_importance_win.csv")

        # Spread model (non-push games)
        mask = y_test_spread_outcome != 0
        df_spread = self.get_feature_importance(
            model=model_spread,
            X_test=X_test_scaled[mask],
            y_test=(y_test_spread_outcome[mask] > 0).astype(int)
        )
        df_spread.to_csv("Feature Importance/feature_importance_spread.csv", index=False)
        print("Saved Spread model importances to feature_importance_spread.csv")

        # Over/Under model (non-push games)
        mask = y_test_total_outcome != 0
        df_over = self.get_feature_importance(
            model=model_over,
            X_test=X_test_scaled[mask],
            y_test=(y_test_total_outcome[mask] > 0).astype(int)
        )
        df_over.to_csv("Feature Importance/feature_importance_overunder.csv", index=False)
        print("Saved Over/Under model importances to feature_importance_overunder.csv")

        self._save_model(self.model_, self.scaler, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) with {len(X_train)} train / {len(X_test)} test games.")
        print("Test predictions and data are now available in instance attributes (e.g., model.predictions).")


    # =================================================================================
    # Internal Helper Methods
    # =================================================================================

    def _select_features(self, X: np.ndarray, random_state: int) -> np.ndarray:
        """
        Selects features based on the initialization settings.
        Priority: 1. feature_allowlist, 2. random_subset, 3. all features.
        """
        # Case 1: A specific list of features is provided
        if self.feature_allowlist:
            print(f"INFO: Filtering features based on the provided allowlist of {len(self.feature_allowlist)} features.")
            
            # Create a mapping from feature name to its index in the full dataset
            name_to_index = {name: i for i, name in enumerate(self.feature_names_)}
            
            found_indices = []
            missing_features = []
            
            for feature_name in self.feature_allowlist:
                if feature_name in name_to_index:
                    found_indices.append(name_to_index[feature_name])
                else:
                    missing_features.append(feature_name)
            
            if missing_features:
                print(f"Warning: The following {len(missing_features)} features from the allowlist were not found and will be ignored: {missing_features}")

            if not found_indices:
                raise ValueError("None of the features in the allowlist were found in the dataset. Aborting.")

            # Sort indices to maintain order and update the feature names attribute
            found_indices.sort()
            self.feature_names_ = [self.feature_names_[i] for i in found_indices]
            self.feature_indices_ = found_indices # Store the selected indices

            print(f"INFO: Using {len(found_indices)} features from the provided allowlist.")
            return X[:, found_indices]

        # Case 2: A random subset of features is requested
        elif self.use_random_subset_of_features:
            n_features = X.shape[1]
            n_selected_features = max(1, int(n_features * self.subset_fraction))

            rng = np.random.default_rng()
            indices = rng.choice(n_features, size=n_selected_features, replace=False)
            indices.sort()
            
            self.feature_indices_ = indices
            # Update feature_names_ to match the selected features
            self.feature_names_ = [self.feature_names_[i] for i in indices]
            print(f"INFO: Using a random subset of {len(self.feature_indices_)} out of {n_features} features.")
            return X[:, self.feature_indices_]

        # Case 3: Default, use all features
        else:
            self.feature_indices_ = None
            return X

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
            for key in self._FEATURE_KEYS_TO_DROP:
                temp_obj.pop(key, None)

            row_features = []
            row_feature_names = []
            
            # Generate feature names only for the first valid JSON object
            if feature_names is None:
                self._flatten_json_to_list(temp_obj, row_features, row_feature_names, generate_names=True)
                feature_names = row_feature_names
            else:
                self._flatten_json_to_list(temp_obj, row_features)

            feature_list.append(row_features)

        if not feature_list:
            return np.array([]), []

        it = iter(feature_list)
        first_row_len = len(next(it, []))
        if not all(len(row) == first_row_len for row in it):
            all_lengths = {len(r) for r in feature_list}
            raise ValueError(
                f"Inconsistent feature lengths detected. All rows must produce a feature vector of the same size. "
                f"Found lengths: {all_lengths}. This suggests the JSON data structure is not consistent across all rows. "
                f"Please fix the upstream data generation process."
            )
        
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
            "model": model,
            "scaler": scaler,
            "model_type": self.model_type,
            "trained_input_shape": trained_input_shape,
            "original_input_shape": original_input_shape,
            "feature_indices": self.feature_indices_,
            "column": self.column,
            "query": query,
            # We don't save feature names in the model file to keep it lean.
            # They are regenerated during the predict phase if needed,
            # but are primarily for interactive analysis after training.
        }
        joblib.dump(model_info, output_path)
        print(f"Model and scaler saved to {output_path}\n")

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
    def _flatten_json_to_list(obj, out_list: list, name_list: list = None, prefix: str = '', generate_names: bool = False):
        """
        Recursively flattens a JSON object (dict/list) into a single list of values
        and optionally generates a corresponding list of feature names.
        """
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                new_prefix = f"{prefix}.{key}" if prefix else key
                MLModel._flatten_json_to_list(obj[key], out_list, name_list, new_prefix, generate_names)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}.{i}"
                MLModel._flatten_json_to_list(v, out_list, name_list, new_prefix, generate_names)
        elif isinstance(obj, bool):
            out_list.append(1.0 if obj else 0.0)
            if generate_names:
                name_list.append(prefix)
        elif isinstance(obj, (int, float)):
            out_list.append(float(obj))
            if generate_names:
                name_list.append(prefix)
        elif obj is None:
            out_list.append(0.0)
            if generate_names:
                name_list.append(prefix)