# predict.py
# Purpose: Load the trained ML model and predict winning probabilities for a given match state.

import pandas as pd
import pickle
import os

def load_predictor_model(model_path):
    """
    Loads the trained machine learning pipeline from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first by running train_model.py.")
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_probability(model, batting_team, bowling_team, city, target, runs_left, balls_left, wickets_left):
    """
    Calculates the current run rate, required run rate, and uses the model 
    to predict win/loss probabilities for both teams.
    """
    # 1. Calculate features that are derived from inputs
    balls_bowled = 120 - balls_left
    current_score = target - runs_left
    
    # Calculate Current Run Rate (CRR)
    if balls_bowled > 0:
        crr = (current_score * 6) / balls_bowled
    else:
        crr = 0.0
        
    # Calculate Required Run Rate (RRR)
    if balls_left > 0:
        rrr = (runs_left * 6) / balls_left
    else:
        rrr = 0.0
        
    # 2. Package the inputs into a Pandas DataFrame
    # Note: Column names must match the exact columns the model was trained on!
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'target': [target],
        'crr': [crr],
        'rrr': [rrr]
    })
    
    # 3. Predict probabilities
    # predict_proba returns an array: [[probability_of_0, probability_of_1]]
    # class 0 = chasing team loses (bowling team wins)
    # class 1 = chasing team wins (batting team wins)
    probabilities = model.predict_proba(input_data)[0]
    
    batting_win_prob = probabilities[1]
    bowling_win_prob = probabilities[0]
    
    return batting_win_prob, bowling_win_prob

if __name__ == "__main__":
    # Test the script locally with dummy data
    MODEL_PATH = "model/model.pkl"
    
    print("Initializing test prediction...")
    try:
        model = load_predictor_model(MODEL_PATH)
        
        # Scenario: Mumbai Indians is chasing Royal Challengers Bangalore's target of 180 runs.
        # Currently, Mumbai Indians is at 120/4 in 15 overs (90 balls bowled, 30 balls left).
        # Mumbai Indians needs 60 runs in 30 balls with 6 wickets in hand.
        batting_team = "Mumbai Indians"
        bowling_team = "Royal Challengers Bangalore"
        city = "Mumbai"
        target = 180
        runs_left = 60
        balls_left = 30
        wickets_left = 6
        
        batting_prob, bowling_prob = predict_probability(
            model, batting_team, bowling_team, city, target, runs_left, balls_left, wickets_left
        )
        
        print("\n--- Test Prediction Result ---")
        print(f"Match: {batting_team} vs {bowling_team} at {city}")
        print(f"Target: {target} | Runs needed: {runs_left} | Balls remaining: {balls_left} | Wickets remaining: {wickets_left}")
        print(f"{batting_team} (Chasing) Win Probability: {batting_prob * 100:.2f}%")
        print(f"{bowling_team} (Defending) Win Probability: {bowling_prob * 100:.2f}%")
        
    except FileNotFoundError as e:
        print(e)
