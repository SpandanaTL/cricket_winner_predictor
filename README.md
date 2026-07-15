# Cricket Match Winner Prediction System 🏏

A complete, end-to-end Machine Learning web application built using **Streamlit** and **Scikit-learn** that predicts the ball-by-ball win probability of the chasing team in an IPL (Indian Premier League) match.

---

## 🎯 Project Objective
The main objective of this project is to build a real-time predictive dashboard that estimates the winning chances of the chasing team. By analyzing historical ball-by-ball matches (2008-2024), our model predicts how factors like wickets remaining, balls left, and required run rate influence the match outcome.

---

## 📊 Dataset
This project uses the historical **IPL Ball-by-Ball and Match Dataset** (freely available on Kaggle/Cricsheet).
* **Matches Dataset (`matches.csv`):** Summary of each match (teams, venue, toss details, winner).
* **Deliveries Dataset (`deliveries.csv`):** Ball-by-ball events of each match (runs scored, extras, wickets, batsman/bowler).

During preprocessing, we combine these datasets to engineer the following key features:
* `batting_team` (The chasing team)
* `bowling_team` (The defending team)
* `city` (Host venue city)
* `runs_left` (Runs needed to reach target)
* `balls_left` (Deliveries remaining out of 120)
* `wickets_left` (Wickets remaining out of 10)
* `target` (Target score set by the 1st innings batting team)
* `crr` (Current Run Rate)
* `rrr` (Required Run Rate)

The **Target Variable** for prediction is `result` (`1` if the batting/chasing team won, `0` if they lost).

---

## 🤖 Machine Learning Algorithm
We use a **Random Forest Classifier** to power our predictions.
* **Why Random Forest?** Cricket situations contain complex, non-linear relationships (e.g., wickets left are much more valuable in the last 3 overs than in the first 3). Random Forest uses an ensemble of multiple Decision Trees to capture these trends accurately.
* **Probability Output:** Instead of outputting a simple "Win/Loss" class, we extract soft probabilities using `predict_proba()`, which maps results to a percentage chance (e.g., 75% Win / 25% Loss).

---

## ⚙️ Project Workflow
Here is how data flows through the application:
1. **Raw CSV Files** (`matches.csv` and `deliveries.csv`) are placed in the `dataset/` directory.
2. **Preprocessing Script** ([preprocess.py](file:///C:/Users/Spandana%20T%20L/.gemini/antigravity/scratch/cricket_winner_predictor/preprocess.py)) cleans the data, handles team name updates, filters second innings, and extracts live match states.
3. **Training Script** ([train_model.py](file:///C:/Users/Spandana%20T%20L/.gemini/antigravity/scratch/cricket_winner_predictor/train_model.py)) converts text data using a One-Hot Encoder, trains the Random Forest classifier on 80% of the data, evaluates it on the remaining 20%, and exports the final model pipeline to `model/model.pkl`.
4. **Streamlit App UI** ([app.py](file:///C:/Users/Spandana%20T%20L/.gemini/antigravity/scratch/cricket_winner_predictor/app.py)) provides an interactive web dashboard for user inputs.
5. **Prediction Module** ([predict.py](file:///C:/Users/Spandana%20T%20L/.gemini/antigravity/scratch/cricket_winner_predictor/predict.py)) calculates the current rates, transforms inputs using the saved model pipeline, and outputs win/loss probabilities.

---

## 🚀 Installation & Running the Project

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Set Up the Directory
Recommend setting the project folder as your active workspace:
`C:\Users\Spandana T L\.gemini\antigravity\scratch\cricket_winner_predictor`

### 2. Install Required Libraries
Open your terminal inside the project directory and run:
```bash
pip install -r requirements.txt
```

### 3. Place Your Datasets
Download the IPL dataset and place the raw files into the `dataset/` folder:
- `dataset/matches.csv`
- `dataset/deliveries.csv`

### 4. Run the Data Preprocessing
Clean and structure the raw match data:
```bash
python preprocess.py
```
*This will generate a clean `dataset/preprocessed_data.csv` file.*

### 5. Train the Model
Train the Random Forest algorithm:
```bash
python train_model.py
```
*This will print evaluation metrics (Accuracy, Precision, Recall) and generate the saved model pipeline inside `model/model.pkl`.*

### 6. Launch the Web Application
Start the interactive Streamlit dashboard:
```bash
streamlit run app.py
```
A new window will open in your web browser at `http://localhost:8501` showing the predictor interface!
