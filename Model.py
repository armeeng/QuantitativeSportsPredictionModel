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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.multioutput import MultiOutputRegressor

# Assuming BaseModel is defined elsewhere, as in the original code.
# If not, it can be a simple pass-through: class BaseModel: def __init__(*args, **kwargs): pass
class BaseModel:
    def __init__(self, model_name: str, column: str):
        self.model_name = model_name
        self.column = column

class MLModel(BaseModel):
    """
    A refactored ML class to train models on sports data.
    
    This class can train regression models to predict scores or a set of
    classification models (e.g., Logistic Regression, KNN) to predict game outcomes
    (win, spread cover, over/under). It now includes an option to train on a
    random subset of features.
    """
    
    # A model factory to map model_type strings to scikit-learn estimators.
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
    # Add aliases
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
            subset_fraction (float): The fraction of features to use if subsetting is enabled (e.g., 0.5 for 50%).
        """
        super().__init__(model_name, column=column)
        self.model_type = model_type.lower()
        if self.model_type not in self._MODELS:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}. Supported types are {list(self._MODELS.keys())}")

        # NEW: Parameters for feature subsetting
        self.use_random_subset_of_features = use_random_subset_of_features
        if subset_fraction is None:
            self.subset_fraction = random.uniform(0.01, 1.0)
            print(f"INFO: No subset_fraction specified. Using a random fraction: {self.subset_fraction:.3f}")
        else:
            if not (0 < subset_fraction <= 1.0):
                raise ValueError("If specified, subset_fraction must be greater than 0 and less than or equal to 1.")
            self.subset_fraction = subset_fraction
        
        # This will store the indices of the features selected during training.
        # The trailing underscore follows scikit-learn convention for learned attributes.
        self.feature_indices_ = None

    # =================================================================================
    # Public API: Main Methods
    # =================================================================================
    
    def train(self, query: str, test_size: float = 0.5, random_state: int = 42):
        """
        Trains and evaluates a model, then saves it to disk.
        Routes to the appropriate training method based on model_type.
        """
        if self.model_type in self._CLASSIFIER_TYPES:
            self._train_classifier(query, test_size, random_state)
        else:
            self._train_regressor(query, test_size, random_state)

    def predict(self, query: str) -> list:
        """
        Loads the latest model and returns predictions for new data.
        Handles both regression (scores) and classification (probabilities) models.
        """
        # 1. Load model and metadata from disk
        try:
            info = self._load_model()
            model = info['model']
            column = info['column']
            # MODIFIED: Load feature selection metadata
            feature_indices = info.get('feature_indices')
            
            # Determine the expected shape of raw features and features for the model
            if 'original_input_shape' in info: # For new models
                original_shape = info['original_input_shape']
                model_input_shape = info['trained_input_shape']
            else: # For backward compatibility with old models
                original_shape = info['input_shape']
                model_input_shape = info['input_shape']
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return []

        # 2. Load and prepare features for new data
        df = self._load_data_from_db(query)
        if df.empty:
            print("Query returned no data to predict on.")
            return []
        
        self.column = column
        X_raw = self._prepare_features(df)

        # 3. Validate raw feature shape before any subsetting
        if X_raw.shape[1] != original_shape:
            raise ValueError(f"Raw feature shape mismatch: model was trained on data with {original_shape} features, but new data has {X_raw.shape[1]}.")

        # 4. Apply the same feature subsetting used during training
        if feature_indices is not None:
            X = X_raw[:, feature_indices]
        else:
            X = X_raw

        # 5. Validate the final feature shape that will be fed to the model
        if X.shape[1] != model_input_shape:
            raise ValueError(f"Final feature shape mismatch: model expects {model_input_shape} features, but got {X.shape[1]}. This can happen if the feature selection logic is inconsistent.")

        # 6. Generate and format predictions
        if isinstance(model, dict):
            # For classifiers, predict probabilities
            prob_win = model['win'].predict_proba(X)[:, 1]
            prob_cover = model['spread'].predict_proba(X)[:, 1]
            prob_over = model['over'].predict_proba(X)[:, 1]
            
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "team1_win_prob": pw, "team1_cover_prob": pc, "over_prob": po}
                for gid, t1, t2, pw, pc, po in zip(df["game_id"], df["team1_id"], df["team2_id"], prob_win, prob_cover, prob_over)
            ]
        else:
            # For regressors, predict team scores
            predictions = model.predict(X)
            return [
                {"game_id": gid, "team1_id": t1, "team2_id": t2, "pred_team1": p1, "pred_team2": p2}
                for gid, t1, t2, (p1, p2) in zip(df["game_id"], df["team1_id"], df["team2_id"], predictions)
            ]

    # =================================================================================
    # Internal Training Methods
    # =================================================================================

    def _train_regressor(self, query: str, test_size: float, random_state: int):
        """Trains a regression model to predict scores."""
        df = self._load_data_from_db(query)
        df.dropna(subset=["team1_score", "team2_score"], inplace=True)
        if df.empty:
            print("Query returned no data to train on. Aborting.")
            return

        # Prepare features and targets
        X_raw = self._prepare_features(df)
        original_feature_count = X_raw.shape[1]
        
        # NEW: Apply feature selection if enabled
        X = self._select_features(X_raw, random_state)
        
        y = df[["team1_score", "team2_score"]].to_numpy()
        
        betting_cols = {
            "spread": df.get("team1_spread"), "total": df.get("total_score"),
            "spr_odds1": df.get("team1_spread_odds"), "spr_odds2": df.get("team2_spread_odds"),
            "over_odds": df.get("over_odds"), "under_odds": df.get("under_odds"),
            "ml1": df.get("team1_moneyline"), "ml2": df.get("team2_moneyline")
        }
        
        # Split data
        data_to_split = [X, y] + [pd.to_numeric(col, errors='coerce').fillna(0) for col in betting_cols.values()]
        splits = train_test_split(*data_to_split, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = splits[0], splits[1], splits[2], splits[3]
        betting_data_test = dict(zip(betting_cols.keys(), splits[5::2]))

        # Train model, evaluate, and save
        model = self._get_model()
        model.fit(X_train, y_train)
        metrics = self._evaluate_model(model, X_test, y_test, betting_data_test)
        self._print_metrics(metrics, len(y_test))
        self._save_model(model, metrics, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) on {len(X_train)} train / {len(X_test)} test games.")

    def _train_classifier(self, query: str, test_size: float, random_state: int):
        """Trains three classification models for win, spread, and total."""
        df = self._load_data_from_db(query)
        df.dropna(subset=["team1_score", "team2_score", "team1_spread", "total_score"], inplace=True)
        if df.empty:
            print("Query returned no data to train on. Aborting.")
            return

        # Prepare features and targets
        X_raw = self._prepare_features(df)
        original_feature_count = X_raw.shape[1]
        
        # NEW: Apply feature selection if enabled
        X = self._select_features(X_raw, random_state)
        
        actual_margin = df["team1_score"] - df["team2_score"]
        actual_total = df["team1_score"] + df["team2_score"]
        y_win = (actual_margin > 0).astype(int)
        y_spread_outcome = actual_margin + pd.to_numeric(df["team1_spread"], errors='coerce').fillna(0)
        y_total_outcome = actual_total - pd.to_numeric(df["total_score"], errors='coerce').fillna(0)
        
        betting_cols = {
            "spread": df.get("team1_spread"), "total": df.get("total_score"),
            "spr_odds1": df.get("team1_spread_odds"), "spr_odds2": df.get("team2_spread_odds"),
            "over_odds": df.get("over_odds"), "under_odds": df.get("under_odds"),
            "ml1": df.get("team1_moneyline"), "ml2": df.get("team2_moneyline")
        }

        # Split all data
        data_to_split = [X, y_win, y_spread_outcome, y_total_outcome, df[["team1_score", "team2_score"]]] + \
                        [pd.to_numeric(col, errors='coerce').fillna(0) for col in betting_cols.values()]
        splits = train_test_split(*data_to_split, test_size=test_size, random_state=random_state, stratify=y_win)
        X_train, X_test = splits[0], splits[1]
        y_train_win, y_test_win = splits[2], splits[3]
        y_train_spread_outcome, _ = splits[4], splits[5]
        y_train_total_outcome, _ = splits[6], splits[7]
        y_test_scores = splits[9]
        betting_data_test = dict(zip(betting_cols.keys(), splits[11::2]))

        # Train three separate models
        model_win = self._get_model().fit(X_train, y_train_win)
        
        train_spread_non_push = y_train_spread_outcome != 0
        X_train_spread = X_train[train_spread_non_push]
        y_train_spread = (y_train_spread_outcome[train_spread_non_push] > 0).astype(int)
        model_spread = self._get_model().fit(X_train_spread, y_train_spread)

        train_total_non_push = y_train_total_outcome != 0
        X_train_total = X_train[train_total_non_push]
        y_train_total = (y_train_total_outcome[train_total_non_push] > 0).astype(int)
        model_over = self._get_model().fit(X_train_total, y_train_total)

        models = {'win': model_win, 'spread': model_spread, 'over': model_over}
        
        # Evaluate and save
        metrics = self._evaluate_model(models, X_test, y_test_scores, betting_data_test)
        self._print_metrics(metrics, len(y_test_scores))
        self._save_model(models, metrics, query, X.shape[1], original_feature_count)
        print(f"Trained {self.model_name} ({self.model_type}) with {len(X_train)} train / {len(X_test)} test games.")

    # =================================================================================
    # Internal Helper Methods
    # =================================================================================
    
    def _select_features(self, X: np.ndarray, random_state: int) -> np.ndarray:
        """
        Selects a random subset of features if the option is enabled during initialization.
        The selected feature indices are stored in `self.feature_indices_`.
        
        Args:
            X (np.ndarray): The input feature matrix.
            random_state (int): Seed for the random number generator for reproducibility.
            
        Returns:
            np.ndarray: The feature matrix, potentially with a subset of columns.
        """
        if self.use_random_subset_of_features:
            n_features = X.shape[1]
            n_selected_features = int(n_features * self.subset_fraction)
            n_selected_features = max(1, n_selected_features)  # Ensure at least one feature is selected

            # Use a seeded random number generator for reproducibility
            #rng = np.random.default_rng(random_state)
            rng = np.random.default_rng()
            indices = rng.choice(n_features, size=n_selected_features, replace=False)
            indices.sort()  # Sort for consistency, although not strictly required
            
            self.feature_indices_ = indices
            print(f"INFO: Using a random subset of {len(self.feature_indices_)} out of {n_features} features.")
            return X[:, self.feature_indices_]
        else:
            self.feature_indices_ = None # Ensure it's None if not used
            return X

    def _load_data_from_db(self, query: str) -> pd.DataFrame:
        """Connects to the database and returns a DataFrame for the given query."""
        with sqlite3.connect("sports.db") as conn:
            return pd.read_sql_query(query, conn)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Parses and flattens a JSON column from a DataFrame into a feature matrix."""
        if self.column not in df.columns:
            raise ValueError(f"Column '{self.column}' not found in the DataFrame.")
        
        parsed_json = df[self.column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        feature_list = []
        for js_obj in parsed_json:
            temp_obj = js_obj.copy()
            for key in self._FEATURE_KEYS_TO_DROP:
                temp_obj.pop(key, None)
            
            row_features = []
            self._flatten_json_to_list(temp_obj, row_features)
            feature_list.append(row_features)
        
        if not feature_list: return np.array([])
        if len(set(len(r) for r in feature_list)) > 1:
            raise ValueError("Inconsistent feature lengths found after processing JSON objects.")
        return np.array(feature_list, dtype=float)

    def _get_model(self):
        """Initializes a model instance from the model factory."""
        return self._MODELS[self.model_type]()

    def _save_model(self, model, metrics: dict, query: str, trained_input_shape: int, original_input_shape: int):
        """Saves the model and metadata to a versioned .joblib file."""
        os.makedirs("models", exist_ok=True)
        base_path = os.path.join("models", self.model_name)
        output_path = f"{base_path}.joblib"
        
        if os.path.exists(output_path):
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            archive_path = f"{base_path}_{timestamp}.joblib"
            os.rename(output_path, archive_path)
            print(f"Archived existing model to {archive_path}")

        # MODIFIED: Save new metadata for feature selection
        model_info = {
            "model": model, 
            "model_type": self.model_type, 
            "trained_input_shape": trained_input_shape,   # Shape model was trained on
            "original_input_shape": original_input_shape, # Shape of raw features before selection
            "feature_indices": self.feature_indices_,     # Indices of selected features
            "column": self.column, 
            "query": query, 
            "metrics": metrics
        }
        joblib.dump(model_info, output_path)
        print(f"Model saved to {output_path}\n")

    def _load_model(self) -> dict:
        """Loads the most recent model file matching the model_name."""
        path = os.path.join("models", f"{self.model_name}.joblib")
        if not os.path.exists(path):
             # Fallback to glob pattern for older versioned files if the main one doesn't exist
            pattern = os.path.join("models", f"{self.model_name}*.joblib")
            candidates = glob.glob(pattern)
            if not candidates:
                raise FileNotFoundError(f"No model files found for '{self.model_name}'")
            latest_file = max(candidates, key=os.path.getmtime)
            print(f"Loading latest versioned model from {latest_file}")
            return joblib.load(latest_file)
        
        print(f"Loading model from {path}")
        return joblib.load(path)

    # =================================================================================
    # Evaluation and Static Helpers (No changes in this section)
    # =================================================================================
    
    @staticmethod
    def _flatten_json_to_list(obj, out_list: list):
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

    @staticmethod
    def _calculate_pnl(pick_condition: np.ndarray, actual_vs_line: np.ndarray, odds: np.ndarray) -> float:
        won = (pick_condition & (actual_vs_line > 0)) | (~pick_condition & (actual_vs_line < 0))
        push = (actual_vs_line == 0)
        win_profit = np.where(odds > 0, odds / 100.0, 100.0 / np.abs(odds, where=np.abs(odds)>0, out=np.full_like(odds, np.nan)))
        pnl = np.where(push, 0.0, np.where(won, win_profit, -1.0))
        valid_bets = (odds != 0) & np.isfinite(odds) & np.isfinite(actual_vs_line)
        return np.sum(pnl[valid_bets])

    @staticmethod
    def _evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, betting_data: dict) -> dict:
        if isinstance(model, dict):
            pick_t1_win = model['win'].predict(X_test).astype(bool)
            pick_t1_cover = model['spread'].predict(X_test).astype(bool)
            pick_over = model['over'].predict(X_test).astype(bool)
            actual_margin = y_test.iloc[:, 0] - y_test.iloc[:, 1]
            actual_total = y_test.sum(axis=1)
            win_acc = accuracy_score((actual_margin > 0), pick_t1_win)
            line_spread = pd.to_numeric(betting_data['spread'], errors='coerce').fillna(0)
            actual_vs_line_spread = actual_margin + line_spread
            valid_spread = (actual_vs_line_spread != 0) & np.isfinite(line_spread)
            spread_acc = accuracy_score((actual_vs_line_spread[valid_spread] > 0), pick_t1_cover[valid_spread]) if valid_spread.any() else np.nan
            spread_odds = np.where(pick_t1_cover, betting_data['spr_odds1'], betting_data['spr_odds2'])
            spread_pnl = MLModel._calculate_pnl(pick_t1_cover, actual_vs_line_spread, spread_odds)
            spread_games_counted = np.sum((spread_odds != 0) & np.isfinite(spread_odds) & np.isfinite(actual_vs_line_spread))
            line_total = pd.to_numeric(betting_data['total'], errors='coerce').fillna(0)
            actual_vs_line_total = actual_total - line_total
            valid_ou = (actual_vs_line_total != 0) & np.isfinite(line_total)
            ou_acc = accuracy_score((actual_vs_line_total[valid_ou] > 0), pick_over[valid_ou]) if valid_ou.any() else np.nan
            ou_odds = np.where(pick_over, betting_data['over_odds'], betting_data['under_odds'])
            ou_pnl = MLModel._calculate_pnl(pick_over, actual_vs_line_total, ou_odds)
            ou_games_counted = np.sum((ou_odds != 0) & np.isfinite(ou_odds) & np.isfinite(actual_vs_line_total))
            ml_odds = np.where(pick_t1_win, betting_data['ml1'], betting_data['ml2'])
            ml_pnl = MLModel._calculate_pnl(pick_t1_win, actual_margin, ml_odds)
            ml_games_counted = np.sum((ml_odds != 0) & np.isfinite(ml_odds) & np.isfinite(actual_margin))
            return {"win_acc": win_acc, "spread_acc": spread_acc, "ou_acc": ou_acc, "ml_pnl": ml_pnl, "spread_pnl": spread_pnl, "ou_pnl": ou_pnl, "ml_games_counted": ml_games_counted, "spread_games_counted": spread_games_counted, "ou_games_counted": ou_games_counted}
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        actual_margin = y_test[:, 0] - y_test[:, 1]
        pred_margin = y_pred[:, 0] - y_pred[:, 1]
        actual_total = y_test.sum(axis=1)
        pred_total = y_pred.sum(axis=1)
        win_acc = np.mean((pred_margin > 0) == (actual_margin > 0))
        line_spread = pd.to_numeric(betting_data['spread'], errors='coerce').fillna(0)
        actual_vs_line_spread = actual_margin + line_spread
        valid_spread = (actual_vs_line_spread != 0) & np.isfinite(line_spread)
        pick_t1_cover = (pred_margin + line_spread) > 0
        spread_acc = np.mean(pick_t1_cover[valid_spread] == (actual_vs_line_spread[valid_spread] > 0)) if valid_spread.any() else np.nan
        spread_odds = np.where(pick_t1_cover, betting_data['spr_odds1'], betting_data['spr_odds2'])
        spread_pnl = MLModel._calculate_pnl(pick_t1_cover, actual_vs_line_spread, spread_odds)
        spread_games_counted = np.sum((spread_odds != 0) & np.isfinite(spread_odds) & np.isfinite(actual_vs_line_spread))
        line_total = pd.to_numeric(betting_data['total'], errors='coerce').fillna(0)
        actual_vs_line_total = actual_total - line_total
        valid_ou = (actual_vs_line_total != 0) & np.isfinite(line_total)
        pick_over = pred_total > line_total
        ou_acc = np.mean(pick_over[valid_ou] == (actual_vs_line_total[valid_ou] > 0)) if valid_ou.any() else np.nan
        ou_odds = np.where(pick_over, betting_data['over_odds'], betting_data['under_odds'])
        ou_pnl = MLModel._calculate_pnl(pick_over, actual_vs_line_total, ou_odds)
        ou_games_counted = np.sum((ou_odds != 0) & np.isfinite(ou_odds) & np.isfinite(actual_vs_line_total))
        pick_t1_win = pred_margin > 0
        ml_odds = np.where(pick_t1_win, betting_data['ml1'], betting_data['ml2'])
        ml_pnl = MLModel._calculate_pnl(pick_t1_win, actual_margin, ml_odds)
        ml_games_counted = np.sum((ml_odds != 0) & np.isfinite(ml_odds) & np.isfinite(actual_margin))
        return {"r2": r2.tolist(), "rmse": rmse.tolist(), "mae": mae.tolist(), "win_acc": win_acc, "spread_acc": spread_acc, "ou_acc": ou_acc, "ml_pnl": ml_pnl, "spread_pnl": spread_pnl, "ou_pnl": ou_pnl, "ml_games_counted": ml_games_counted, "spread_games_counted": spread_games_counted, "ou_games_counted": ou_games_counted}

    @staticmethod
    def _print_metrics(metrics: dict, test_set_size: int):
        if 'r2' in metrics:
            print(f"\nRegression metrics on {test_set_size} held-out games:")
            print(f"  Team1 → R² {metrics['r2'][0]:.3f},  RMSE {metrics['rmse'][0]:.2f},  MAE {metrics['mae'][0]:.2f}")
            print(f"  Team2 → R² {metrics['r2'][1]:.3f},  RMSE {metrics['rmse'][1]:.2f},  MAE {metrics['mae'][1]:.2f}")
        
        print(f"\nBetting metrics on {test_set_size} games:")
        spread_acc_str = f"{metrics['spread_acc']:.3%}" if pd.notna(metrics['spread_acc']) else "N/A"
        ou_acc_str = f"{metrics['ou_acc']:.3%}" if pd.notna(metrics['ou_acc']) else "N/A"
        ml_games_str = f"(on {metrics.get('ml_games_counted', 'N/A')} games)"
        spread_games_str = f"(on {metrics.get('spread_games_counted', 'N/A')} games)"
        ou_games_str = f"(on {metrics.get('ou_games_counted', 'N/A')} games)"

        print(f"  Win/Loss accuracy:    {metrics['win_acc']:.3%}")
        print(f"  Spread-cover accuracy:{spread_acc_str}")
        print(f"  Over/Under accuracy:  {ou_acc_str}")
        print(f"  Moneyline P/L:        {metrics['ml_pnl']:.2f} units per $1 {ml_games_str}")
        print(f"  Spread P/L:           {metrics['spread_pnl']:.2f} units per $1 {spread_games_str}")
        print(f"  Over/Under P/L:       {metrics['ou_pnl']:.2f} units per $1 {ou_games_str}")