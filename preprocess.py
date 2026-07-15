# preprocess.py
# Purpose: Clean and transform raw IPL matches and deliveries data into features for Machine Learning.
# Automatically downloads data from public GitHub repositories if missing.

import pandas as pd
import numpy as np
import os
import urllib.request

# Public URLs hosting the standard IPL dataset
MATCHES_URLS = [
    "https://raw.githubusercontent.com/Shivaae/IPL-DATA-/master/matches.csv",
    "https://raw.githubusercontent.com/ramakrishnansureshbabu/IPL-Winner-Prediction/master/matches.csv"
]

DELIVERIES_URLS = [
    "https://raw.githubusercontent.com/Shivaae/IPL-DATA-/master/deliveries.csv",
    "https://raw.githubusercontent.com/ramakrishnansureshbabu/IPL-Winner-Prediction/master/deliveries.csv"
]

def download_file(urls, save_path):
    """
    Tries to download a file from a list of URLs. Saves to save_path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for url in urls:
        try:
            print(f"Trying to download from: {url} ...")
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(save_path, 'wb') as out_file:
                    out_file.write(response.read())
            print(f"Successfully downloaded and saved to {save_path}!")
            return
        except Exception as e:
            print(f"Failed to download from this URL. Error: {e}")
            
    raise RuntimeError(f"Could not download the file from any of the URLs. Please download it manually and place it in {save_path}")

def standardize_columns(matches_df, deliveries_df):
    """
    Renames columns in both dataframes to standardize column names.
    This resolves issues when different dataset versions are used.
    """
    print("Standardizing column names across dataframes...")
    
    # 1. Standardize Deliveries columns
    # If match identifier in deliveries is named 'id' instead of 'match_id', rename it
    if 'match_id' not in deliveries_df.columns and 'id' in deliveries_df.columns:
        deliveries_df = deliveries_df.rename(columns={'id': 'match_id'})
        print("-> Renamed deliveries column 'id' to 'match_id'")
        
    # Standardize 'innings' to 'inning'
    if 'inning' not in deliveries_df.columns and 'innings' in deliveries_df.columns:
        deliveries_df = deliveries_df.rename(columns={'innings': 'inning'})
        print("-> Renamed deliveries column 'innings' to 'inning'")
        
    # Handle total runs variants
    if 'total_runs' not in deliveries_df.columns:
        runs_col = 'batsman_runs' if 'batsman_runs' in deliveries_df.columns else 'runs_off_bat'
        extras_col = 'extra_runs' if 'extra_runs' in deliveries_df.columns else 'extras'
        
        if runs_col in deliveries_df.columns and extras_col in deliveries_df.columns:
            deliveries_df['total_runs'] = deliveries_df[runs_col] + deliveries_df[extras_col]
            print(f"-> Calculated 'total_runs' from '{runs_col}' and '{extras_col}'")
            
    # 2. Standardize Matches columns
    # Ensure matches has 'id' representing the match ID
    if 'id' not in matches_df.columns and 'match_id' in matches_df.columns:
        matches_df = matches_df.rename(columns={'match_id': 'id'})
        print("-> Renamed matches column 'match_id' to 'id'")
        
    # Verify critical columns exist
    required_matches_cols = ['id', 'city', 'winner', 'team1', 'team2']
    required_deliveries_cols = ['match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball', 'total_runs']
    
    for col in required_matches_cols:
        if col not in matches_df.columns:
            raise KeyError(f"Required column '{col}' is missing in matches.csv")
            
    for col in required_deliveries_cols:
        if col not in deliveries_df.columns:
            raise KeyError(f"Required column '{col}' is missing in deliveries.csv")
            
    return matches_df, deliveries_df

def load_data(matches_path, deliveries_path):
    """
    Loads matches and deliveries CSV files.
    """
    if not os.path.exists(matches_path):
        raise FileNotFoundError(f"Matches file not found at: {matches_path}")
    if not os.path.exists(deliveries_path):
        raise FileNotFoundError(f"Deliveries file not found at: {deliveries_path}")
        
    print("Loading datasets...")
    matches_df = pd.read_csv(matches_path)
    deliveries_df = pd.read_csv(deliveries_path)
    print(f"Loaded matches shape: {matches_df.shape}")
    print(f"Loaded deliveries shape: {deliveries_df.shape}")
    return matches_df, deliveries_df

def clean_data(matches_df, deliveries_df):
    """
    Filters matches to keep only completed matches, standardizes team names,
    and returns a clean merge of match outcomes with ball-by-ball events.
    """
    print("Cleaning and merging datasets...")
    
    # 1. Map old/defunct team names to current active franchise names
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Deccan Chargers': 'Sunrisers Hyderabad',
        'Rising Pune Supergiants': 'Rising Pune Supergiant'
    }
    
    # Define the list of primary active teams we want to keep
    active_teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Punjab Kings', 'Chennai Super Kings',
        'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Titans', 'Lucknow Super Giants'
    ]
    
    # Apply team mapping in matches
    matches_df['team1'] = matches_df['team1'].replace(team_mapping)
    matches_df['team2'] = matches_df['team2'].replace(team_mapping)
    matches_df['winner'] = matches_df['winner'].replace(team_mapping)
    
    # Apply team mapping in deliveries
    deliveries_df['batting_team'] = deliveries_df['batting_team'].replace(team_mapping)
    deliveries_df['bowling_team'] = deliveries_df['bowling_team'].replace(team_mapping)
    
    # Filter matches to only include matches between active teams
    matches_df = matches_df[matches_df['team1'].isin(active_teams)]
    matches_df = matches_df[matches_df['team2'].isin(active_teams)]
    
    # Filter deliveries to match active teams
    deliveries_df = deliveries_df[deliveries_df['batting_team'].isin(active_teams)]
    deliveries_df = deliveries_df[deliveries_df['bowling_team'].isin(active_teams)]
    
    # 2. Filter matches: Keep only completed matches and ignore rain-affected matches (No DL rule)
    if 'dl_applied' in matches_df.columns:
        matches_df = matches_df[matches_df['dl_applied'] == 0]
    
    # Keep only matches that have a clear winner
    matches_df = matches_df.dropna(subset=['winner'])
    
    # 3. Get the first innings total runs for each match to calculate targets
    total_score_df = deliveries_df.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    
    # We only need the score of the 1st innings to set the target for the 2nd innings
    first_innings_score = total_score_df[total_score_df['inning'] == 1]
    
    # Target = 1st innings score + 1
    first_innings_score = first_innings_score.rename(columns={'total_runs': 'target_score'})
    first_innings_score['target'] = first_innings_score['target_score'] + 1
    
    # Merge matches metadata and target score back into deliveries
    matches_subset = matches_df[['id', 'city', 'winner']].rename(columns={'id': 'match_id'})
    
    merged_df = deliveries_df.merge(matches_subset, on='match_id', how='inner')
    merged_df = merged_df.merge(first_innings_score[['match_id', 'target']], on='match_id', how='inner')
    
    # 4. We only need the 2nd innings (chase) to predict the outcome live
    merged_df = merged_df[merged_df['inning'] == 2]
    
    print(f"Cleaned and merged shape (2nd innings only): {merged_df.shape}")
    return merged_df

def extract_features(merged_df):
    """
    Extracts the key metrics (features) needed for predictions at every ball.
    """
    print("Extracting features...")
    
    # Sort by match_id, over, and ball to ensure time-sequence order
    merged_df = merged_df.sort_values(by=['match_id', 'over', 'ball'])
    
    # 1. Current Score: Cumulative sum of total_runs scored in the 2nd innings
    merged_df['current_score'] = merged_df.groupby('match_id')['total_runs'].cumsum()
    
    # 2. Runs Left to Win: Target - Current Score
    merged_df['runs_left'] = merged_df['target'] - merged_df['current_score']
    merged_df['runs_left'] = merged_df['runs_left'].apply(lambda x: max(0, x))
    
    # 3. Balls Left: Calculate balls bowled so far and subtract from 120
    min_over = merged_df['over'].min()
    if min_over == 0:
        # Over is 0-19.
        merged_df['balls_bowled'] = (merged_df['over'] * 6) + merged_df['ball']
    else:
        # Over is 1-20.
        merged_df['balls_bowled'] = ((merged_df['over'] - 1) * 6) + merged_df['ball']
        
    merged_df['balls_left'] = 120 - merged_df['balls_bowled']
    merged_df['balls_left'] = merged_df['balls_left'].apply(lambda x: max(0, min(120, x)))
    
    # 4. Wickets Left: Calculate wickets fallen so far and subtract from 10
    # Handle cases where player_dismissed is missing or named slightly differently
    dismissed_col = 'player_dismissed' if 'player_dismissed' in merged_df.columns else 'dismissed_player'
    if dismissed_col in merged_df.columns:
        merged_df['is_wicket'] = merged_df[dismissed_col].apply(lambda x: 0 if pd.isna(x) or str(x).strip() == "" or x == 0 else 1)
    else:
        # Fallback if no dismissal details found (set wickets left to default 10)
        merged_df['is_wicket'] = 0
        
    wickets_fallen = merged_df.groupby('match_id')['is_wicket'].cumsum()
    merged_df['wickets_left'] = 10 - wickets_fallen
    merged_df['wickets_left'] = merged_df['wickets_left'].apply(lambda x: max(0, min(10, x)))
    
    # 5. Current Run Rate (CRR): (Current Score * 6) / Balls Bowled
    merged_df['crr'] = np.where(merged_df['balls_bowled'] > 0, 
                                (merged_df['current_score'] * 6) / merged_df['balls_bowled'], 
                                0)
    
    # 6. Required Run Rate (RRR): (Runs Left * 6) / Balls Left
    merged_df['rrr'] = np.where(merged_df['balls_left'] > 0, 
                                (merged_df['runs_left'] * 6) / merged_df['balls_left'], 
                                np.where(merged_df['runs_left'] > 0, 36.0, 0.0))
    
    # 7. Target Variable (Result): 1 if the batting team won, else 0
    def check_winner(row):
        return 1 if row['batting_team'] == row['winner'] else 0
        
    merged_df['result'] = merged_df.apply(check_winner, axis=1)
    
    # Keep only the columns we actually need for the Machine Learning model
    final_columns = [
        'batting_team',
        'bowling_team',
        'city',
        'runs_left',
        'balls_left',
        'wickets_left',
        'target',
        'crr',
        'rrr',
        'result'
    ]
    
    preprocessed_df = merged_df[final_columns]
    preprocessed_df = preprocessed_df.dropna()
    
    # Shuffle the final data
    preprocessed_df = preprocessed_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final preprocessed data shape: {preprocessed_df.shape}")
    return preprocessed_df

def run_preprocessing(matches_path, deliveries_path, output_path):
    """
    Main function to run the full preprocessing pipeline.
    Downloads raw files if they are missing.
    """
    # Check if raw files exist, download if they don't
    if not os.path.exists(matches_path):
        print(f"Raw match dataset is missing. Downloading...")
        download_file(MATCHES_URLS, matches_path)
        
    if not os.path.exists(deliveries_path):
        print(f"Raw deliveries dataset is missing. Downloading...")
        download_file(DELIVERIES_URLS, deliveries_path)
        
    matches_df, deliveries_df = load_data(matches_path, deliveries_path)
    
    # Standardize column variations before executing clean and features steps
    matches_df, deliveries_df = standardize_columns(matches_df, deliveries_df)
    
    cleaned_df = clean_data(matches_df, deliveries_df)
    preprocessed_df = extract_features(cleaned_df)
    
    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preprocessed_df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset successfully saved to: {output_path}")

if __name__ == "__main__":
    MATCHES_CSV = "dataset/matches.csv"
    DELIVERIES_CSV = "dataset/deliveries.csv"
    OUTPUT_CSV = "dataset/processed_data.csv"
    
    print("Starting preprocessing pipeline...")
    run_preprocessing(MATCHES_CSV, DELIVERIES_CSV, OUTPUT_CSV)
