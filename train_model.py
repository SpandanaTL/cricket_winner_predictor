# train_model.py
# Purpose: Train a Random Forest Classifier on preprocessed cricket data and save the model pipeline.

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate(data_path, model_save_path):
    """
    Loads preprocessed data, builds an ML pipeline, trains the Random Forest,
    evaluates its performance, and saves it.
    """
    # 1. Load the preprocessed dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Preprocessed data file not found at: {data_path}. Please run preprocess.py first.")
        
    print("Loading preprocessed dataset...")
    df = pd.read_csv(data_path)
    
    # 2. Split dataset into Features (X) and Target (y)
    # X contains variables we use to predict (inputs)
    # y contains the variable we want to predict (output: 1 for win, 0 for loss)
    X = df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'target', 'crr', 'rrr']]
    y = df['result']
    
    # 3. Split into Training (80%) and Testing (20%) sets
    # We train the model on X_train/y_train, and check how good it is using X_test/y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size: {X_test.shape[0]} rows")
    
    # 4. Handle Categorical Columns (batting_team, bowling_team, city)
    # ML models only understand numbers. One-Hot Encoding converts text categories into binary columns (0s and 1s).
    # For example, "Mumbai Indians" is converted into columns like is_MI=1, is_CSK=0, is_RCB=0.
    categorical_features = ['batting_team', 'bowling_team', 'city']
    
    # Create the column transformer which applies OneHotEncoder only to the categorical columns
    # 'passthrough' means the numerical columns (runs_left, crr, etc.) will not be touched
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 5. Create a Machine Learning Pipeline
    # A pipeline binds the preprocessing step and the model together into a single object.
    # When we feed raw data to this pipeline later, it will automatically encode the text columns and then predict.
    # We use a Random Forest Classifier: an ensemble of decision trees that works great on tabular data.
    # We set max_depth=10 to keep the model size compact and avoid overfitting.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])
    
    # 6. Train the model
    print("Training the Random Forest Classifier... (This might take a moment)")
    model_pipeline.fit(X_train, y_train)
    print("Training completed!")
    
    # 7. Evaluate the model on the test dataset
    print("\n--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy (percentage of correct predictions): {accuracy * 100:.2f}%")
    print(f"Precision (when it predicts a win, how often is it right?): {precision:.2f}")
    print(f"Recall (how many of the actual wins did it catch?): {recall:.2f}")
    print(f"F1 Score (balance between precision and recall): {f1:.2f}")
    print("\nConfusion Matrix:")
    print("Columns: Predicted Loss | Predicted Win")
    print("Rows:    Actual Loss    | Actual Win")
    print(conf_matrix)
    
    # 8. Save the trained model pipeline
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as file:
        pickle.dump(model_pipeline, file)
    print(f"\nTrained model pipeline successfully saved to: {model_save_path}")

if __name__ == "__main__":
    DATA_PATH = "dataset/processed_data.csv"
    MODEL_SAVE_PATH = "model/model.pkl"
    train_and_evaluate(DATA_PATH, MODEL_SAVE_PATH)
