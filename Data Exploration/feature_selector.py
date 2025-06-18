import csv
import argparse

def extract_important_indices(file_path, num_features, match_stats=False):
    """
    Extracts the indices of the most important features from a CSV file.

    The input CSV is expected to have 'feature_index', 'feature', and 'importance' columns.

    Args:
        file_path (str): The path to the CSV file.
        num_features (int): The number of top features to consider.
        match_stats (bool): If True, ensures that if a stat for team1 is selected,
                            the corresponding stat for team2 is also included, and vice-versa.

    Returns:
        list: A sorted list of the most important feature indices.
    """
    all_features = []
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    # Read the index, name, and importance for each feature
                    all_features.append({
                        'index': int(row['feature_index']),
                        'feature': row['feature'],
                        'importance': float(row['importance'])
                    })
                except (ValueError, KeyError) as e:
                    print(f"Skipping row due to malformed data: {row} - Error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    # Sort all features by importance in descending order
    all_features.sort(key=lambda x: x['importance'], reverse=True)

    # Get the top N feature dictionaries based on importance
    top_feature_dicts = all_features[:num_features]

    if not match_stats:
        # If not matching, simply return the indices of the top N features
        return [f['index'] for f in top_feature_dicts]
    else:
        # If matching, perform the counterpart logic
        # Create lookup maps from the full feature list for efficiency
        all_feature_names = {f['feature'] for f in all_features}
        name_to_index_map = {f['feature']: f['index'] for f in all_features}
        
        # Start with a set of the indices from the top N features
        final_indices = {f['index'] for f in top_feature_dicts}
        
        # Iterate through the top features to find and add their counterparts
        for feature_dict in top_feature_dicts:
            feature_name = feature_dict['feature']
            
            if feature_name.startswith('team1_stats.'):
                stat_name = feature_name.replace('team1_stats.', '')
                counterpart = f'team2_stats.{stat_name}'
                # If the counterpart exists, add its index to the set
                if counterpart in all_feature_names:
                    final_indices.add(name_to_index_map[counterpart])

            elif feature_name.startswith('team2_stats.'):
                stat_name = feature_name.replace('team2_stats.', '')
                counterpart = f'team1_stats.{stat_name}'
                # If the counterpart exists, add its index to the set
                if counterpart in all_feature_names:
                    final_indices.add(name_to_index_map[counterpart])
        
        # Return a sorted list of the unique indices
        return sorted(list(final_indices))

if __name__ == "__main__":
    # --- Configuration for Feature Extraction ---
    # Set the path to your CSV file
    csv_file_path = 'Feature Importance/feature_importance_spread.csv' 
    
    # Set the number of top features you want to extract
    num_features_to_extract = 100
    
    # Set to True if you want to ensure consistency between team1 and team2 stats
    should_match_stats = True
    # ------------------------------------------

    # The function now returns indices, so the variable name is updated for clarity
    important_indices = extract_important_indices(
        csv_file_path,
        num_features_to_extract,
        should_match_stats
    )

    if important_indices:
        # The output format is updated to reflect that it's a list of indices
        print(f"indices = {important_indices}")
    else:
        print("No feature indices extracted. Please check file path and content.")