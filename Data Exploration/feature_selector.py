import csv
import argparse

def extract_important_features(file_path, num_features, match_stats=False):
    """
    Extracts the most important features from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing feature importance.
        num_features (int): The number of top features to extract.
        match_stats (bool): If True, ensures that if 'team1_stats.X' is selected,
                            'team2_stats.X' is also included (if present in the CSV),
                            and vice-versa, for consistency.

    Returns:
        list: A list of the most important features.
    """
    all_features = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    feature = row['feature']
                    importance = float(row['importance'])
                    all_features.append({'feature': feature, 'importance': importance})
                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to malformed data: {row} - Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    # Sort features by importance in descending order
    all_features.sort(key=lambda x: x['importance'], reverse=True)

    # Create a dictionary for quick lookup of all features
    all_feature_names = {f['feature'] for f in all_features}
    
    # Get the top N features based on importance
    top_features = [f['feature'] for f in all_features[:num_features]]

    if match_stats:
        final_features = set(top_features) # Use a set to avoid duplicates
        
        for feature in top_features:
            if feature.startswith('team1_stats.'):
                stat_name = feature.replace('team1_stats.', '')
                counterpart = f'team2_stats.{stat_name}'
                if counterpart in all_feature_names:
                    final_features.add(counterpart)
            elif feature.startswith('team2_stats.'):
                stat_name = feature.replace('team2_stats.', '')
                counterpart = f'team1_stats.{stat_name}'
                if counterpart in all_feature_names:
                    final_features.add(counterpart)
        
        return sorted(list(final_features)) # Return sorted list
    else:
        return top_features

if __name__ == "__main__":
    # --- Configuration for Feature Extraction ---
    # Set the path to your CSV file
    csv_file_path = 'Feature Importance/feature_importance_spread.csv' 
    
    # Set the number of top features you want to extract
    num_features_to_extract = 100
    
    # Set to True if you want to ensure consistency between team1 and team2 stats
    # (e.g., if 'team1_stats.X' is selected, 'team2_stats.X' will also be included if present)
    should_match_stats = True
    # ------------------------------------------

    important_features = extract_important_features(
        csv_file_path,
        num_features_to_extract,
        should_match_stats
    )

    if important_features:
        print(f"features = {important_features}")
    else:
        print("No features extracted. Please check file path and content.")