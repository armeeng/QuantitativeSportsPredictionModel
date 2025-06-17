import os
import json
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalyzer:
    """
    A class dedicated to analyzing feature correlations from a database.

    This class loads data, prepares a feature matrix from a JSON column,
    and then calculates, visualizes, and filters a correlation matrix
    to identify and remove highly correlated features (multicollinearity).
    """

    # These keys will be removed from the JSON object before flattening
    _FEATURE_KEYS_TO_DROP = {
        "team1_id", "team2_id", "venue_id", "season_type", "day", "month",
        "year", "days_since_epoch", "game_time", "day_of_week"
    }

    def __init__(self, db_path: str, query: str, json_column: str):
        """
        Initializes the CorrelationAnalyzer.

        Args:
            db_path (str): The file path to the SQLite database.
            query (str): The SQL query to execute to fetch the data.
            json_column (str): The name of the column containing the JSON features.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found at: {db_path}")

        self.db_path = db_path
        self.query = query
        self.json_column = json_column
        self.feature_df = None

    def _load_and_prepare_data(self) -> bool:
        """
        Loads data from the database and prepares the feature matrix.
        
        Returns:
            bool: True if data was loaded and prepared successfully, False otherwise.
        """
        print("Loading and preparing data...")
        try:
            with sqlite3.connect(self.db_path) as conn:
                raw_df = pd.read_sql_query(self.query, conn)
        except Exception as e:
            print(f"Error loading data from the database: {e}")
            return False

        if raw_df.empty:
            print("Query returned no data. Cannot perform analysis.")
            return False
        
        X_raw, feature_names = self._prepare_features(raw_df)
        
        if X_raw.size == 0:
            print("Feature preparation resulted in an empty dataset.")
            return False
            
        self.feature_df = pd.DataFrame(X_raw, columns=feature_names)
        print(f"Data prepared successfully. Shape of feature DataFrame: {self.feature_df.shape}")
        return True

    def _prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, list]:
        """
        Parses and flattens a JSON column into a feature matrix and feature names.
        This is a modified version of the method from the MLModel class.
        """
        if self.json_column not in df.columns:
            raise ValueError(f"Column '{self.json_column}' not found in the DataFrame.")

        parsed_json = df[self.json_column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        feature_list = []
        feature_names = None

        for idx, js_obj in enumerate(parsed_json):
            if js_obj is None:
                continue
            
            temp_obj = js_obj.copy()
            for key in self._FEATURE_KEYS_TO_DROP:
                temp_obj.pop(key, None)

            row_features, row_feature_names = [], []
            
            if feature_names is None:
                self._flatten_json_to_list(temp_obj, row_features, row_feature_names, generate_names=True)
                feature_names = row_feature_names
            else:
                self._flatten_json_to_list(temp_obj, row_features)

            feature_list.append(row_features)

        if not feature_list or not feature_names:
            return np.array([]), []

        # Ensure all feature rows have consistent length
        first_len = len(feature_list[0])
        if not all(len(row) == first_len for row in feature_list):
             raise ValueError("Inconsistent feature lengths detected across rows.")

        return np.array(feature_list, dtype=float), feature_names

    @staticmethod
    def _flatten_json_to_list(obj, out_list: list, name_list: list = None, prefix: str = '', generate_names: bool = False):
        """Recursively flattens a JSON object."""
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                new_prefix = f"{prefix}.{key}" if prefix else key
                CorrelationAnalyzer._flatten_json_to_list(obj[key], out_list, name_list, new_prefix, generate_names)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_prefix = f"{prefix}.{i}"
                CorrelationAnalyzer._flatten_json_to_list(v, out_list, name_list, new_prefix, generate_names)
        elif isinstance(obj, bool):
            out_list.append(1.0 if obj else 0.0)
            if generate_names: name_list.append(prefix)
        elif isinstance(obj, (int, float)):
            out_list.append(float(obj))
            if generate_names: name_list.append(prefix)
        elif obj is None:
            out_list.append(0.0)
            if generate_names: name_list.append(prefix)

    def analyze_and_filter(self, threshold: float = 0.95):
        """
        Calculates correlation, identifies features to drop, and plots a heatmap.

        Args:
            threshold (float): The correlation threshold above which to drop features.
        """
        if self.feature_df is None:
            if not self._load_and_prepare_data():
                return

        print(f"\nCalculating correlation matrix and identifying features with correlation > {threshold}...")
        
        # 1. Calculate the absolute correlation matrix
        corr_matrix = self.feature_df.corr().abs()

        # 2. Iteratively identify and remove correlated features
        to_drop = set()
        columns = corr_matrix.columns
        for i in range(len(columns)):
            if columns[i] in to_drop:
                continue
            for j in range(i + 1, len(columns)):
                if columns[j] in to_drop:
                    continue
                if corr_matrix.iloc[i, j] > threshold:
                    to_drop.add(columns[j])
        
        to_drop = list(to_drop)
        final_features = self.feature_df.columns.tolist()

        # 3. Plot the original heatmap before filtering
        #print("\nDisplaying the full correlation heatmap...")
        #self._plot_heatmap(self.feature_df.corr(), "Full Feature Correlation Matrix")

        # 4. Process the results
        if not to_drop:
            print("\nNo features found with a correlation greater than the threshold.")
        else:
            print(f"\nFound and removed {len(to_drop)} features to resolve multicollinearity: {to_drop}")
            
            # Drop the identified features and plot the new heatmap
            df_filtered = self.feature_df.drop(columns=to_drop)
            final_features = df_filtered.columns.tolist() # Get the final list of features
            
            print(f"\nOriginal shape: {self.feature_df.shape}")
            print(f"Shape after dropping columns: {df_filtered.shape}")
            #print("\nDisplaying the filtered correlation heatmap...")
            #self._plot_heatmap(df_filtered.corr(), f"Filtered Correlation Matrix (Threshold > {threshold})")

        # 5. Print the final list of features
        print("\n" + "="*50)
        print("Final list of features after filtering:")
        print("="*50)
        print(f"features = {final_features}")
        print("\nThis list can be used as a 'feature_allowlist' in the MLModel class.")


    @staticmethod
    def _plot_heatmap(corr_matrix: pd.DataFrame, title: str):
        """Generates and displays a heatmap for the given correlation matrix."""
        # Hide annotations if the matrix is too large to read
        show_annot = corr_matrix.shape[0] < 50 
        
        plt.figure(figsize=(16, 12))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=show_annot, fmt=".2f", linewidths=.5)
        plt.title(title, fontsize=16)
        plt.show()


if __name__ == '__main__':
    # --- Configuration ---
    DB_FILE_PATH = "sports.db" # UPDATE if your database has a different name
    JSON_DATA_COLUMN = "stats"  # The column with the feature JSON
    
    # Use a query to get a representative sample of your data
    # Using a LIMIT is a good idea for a quick analysis. Remove it for a full analysis.
    QUERY_FOR_ANALYSIS = f"SELECT * FROM games WHERE sport = 'MLB';"
    
    # --- Run Analysis ---
    analyzer = CorrelationAnalyzer(
        db_path=DB_FILE_PATH,
        query=QUERY_FOR_ANALYSIS,
        json_column=JSON_DATA_COLUMN
    )
    
    # Define the correlation threshold for filtering
    CORRELATION_THRESHOLD = 0.5
    analyzer.analyze_and_filter(threshold=CORRELATION_THRESHOLD)
