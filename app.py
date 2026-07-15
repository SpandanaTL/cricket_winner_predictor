# app.py
# Purpose: Streamlit application providing a modern, interactive web UI for predicting cricket match winners.
# Overhauled with a custom professional cricket theme and a local base64-encoded full-screen background image.

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import base64
import matplotlib.pyplot as plt

# Set page configuration for a professional look and feel
st.set_page_config(
    page_title="Cricket Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 1. Background image configuration and Base64 loader to bypass browser local security blocks
bg_image_path = "static/stadium.jpg"
encoded_string = ""

# Load the local image and convert to Base64 if it exists
if os.path.exists(bg_image_path):
    try:
        with open(bg_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
    except Exception as e:
        pass

# 2. Main CSS stylesheet string (Note: Standard string block to avoid f-string NameErrors)
css_content = """
<style>
    /* Import Poppins Font from Google */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

    /* Global typography rules - Enforce Poppins throughout the app */
    html, body, [class*="css"], [class*="st-"], p, span, div, label, input, button {
        font-family: 'Poppins', sans-serif !important;
    }

    /* Main container styling with custom dark transparent overlay (rgba(0,0,0,0.6)) and background settings */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    BG_PLACEHOLDER;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #ffffff;
    }

    /* Premium centered title with gold underline */
    .premium-title {
        text-align: center;
        color: #ffffff !important;
        font-size: 2.3rem;
        font-weight: 800;
        margin-top: 15px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .gold-underline {
        width: 140px;
        height: 4px;
        background-color: #FFD54F; /* Gold color */
        margin: 0 auto 35px auto;
        border-radius: 2px;
        box-shadow: 0 0 10px rgba(255, 213, 79, 0.6);
    }

    /* All widget input labels set to pure white and bold */
    div[data-testid="stWidgetLabel"] p, label {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8);
        margin-bottom: 8px !important;
    }

    /* Style slider text elements (ranges, active numbers) in white */
    div[data-testid="stSlider"] * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Dropdown selectboxes styled with white background, black text, and rounded corners */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        padding: 2px 6px !important;
    }
    
    /* Dropdown values set to black text */
    .stSelectbox div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    /* Option item selection lists when expanded */
    div[role="listbox"], div[role="listbox"] ul, div[role="listbox"] li {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
    div[role="listbox"] li:hover {
        background-color: #f1f5f9 !important;
        color: #000000 !important;
    }

    /* Make number input boxes have white background, black text, rounded corners, padding, and shadow */
    .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* Plus and minus controls inside number inputs */
    div[data-testid="stNumberInput"] button {
        background-color: #e2e8f0 !important;
        color: #000000 !important;
        border-radius: 6px !important;
        border: none !important;
    }

    /* Glassmorphism input container card styling - Cricket Green theme */
    .card {
        background: rgba(27, 94, 32, 0.25) !important; /* Cricket Green (#1B5E20) transparency */
        border: 1px solid rgba(255, 213, 79, 0.25) !important; /* Gold outline */
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
    }
    
    .card-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 18px;
        color: #FFD54F !important; /* Gold */
        border-bottom: 1.5px solid rgba(255, 213, 79, 0.3);
        padding-bottom: 8px;
    }

    /* Premium glassmorphism result card - Blue and Gold theme */
    .result-card {
        background: rgba(21, 101, 192, 0.35) !important; /* Blue (#1565C0) transparency */
        border: 2px solid #FFD54F !important; /* Gold border outline */
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 12px 40px 0 rgba(255, 213, 79, 0.2) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        margin-bottom: 24px !important;
    }
    
    /* Win probability percentage outputs */
    .probability-display {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin: 5px 0;
        background: linear-gradient(90deg, #FFD54F 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Predict Button (Green to blue gradient, gold border, glow & zoom animations) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1b5e20 0%, #1565c0 100%) !important; /* Cricket Green to Blue */
        color: #ffffff !important;
        border: 1px solid #FFD54F !important; /* Gold border */
        padding: 14px 28px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(21, 101, 192, 0.35) !important;
        cursor: pointer !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    div.stButton > button:first-child:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 10px 25px rgba(255, 213, 79, 0.45) !important; /* Gold glow on hover */
        background: linear-gradient(90deg, #145318 0%, #0d47a1 100%) !important;
    }
    
    div.stButton > button:first-child:active {
        transform: scale(0.97) !important;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.75);
        font-size: 0.95rem;
        font-weight: 500;
        padding: 24px 0;
        margin-top: 50px;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
        background: rgba(11, 61, 46, 0.85);
        backdrop-filter: blur(4px);
    }
</style>
"""

# Replace the BG placeholder inside the main CSS block using string replacement
if encoded_string:
    css_content = css_content.replace("BG_PLACEHOLDER", f'url("data:image/jpg;base64,{encoded_string}")')
else:
    css_content = css_content.replace("BG_PLACEHOLDER", 'linear-gradient(rgba(11, 61, 46, 0.95), rgba(11, 61, 46, 0.95))')

# Inject the compiled styling in exactly ONE markdown block
st.markdown(css_content, unsafe_allow_html=True)

# Helper to load the trained model pipeline
@st.cache_resource
def load_model():
    model_path = "model/model.pkl"
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    return None

# Load the model
model = load_model()

# Header Section (Centered premium title with gold underline)
st.markdown('<div class="premium-title">🏏 Cricket Match Winner Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="gold-underline"></div>', unsafe_allow_html=True)

# Display a clear Streamlit warning if stadium.jpg is missing in the static directory
if not os.path.exists(bg_image_path):
    st.warning("⚠️ Background Warning: Local background image 'static/stadium.jpg' was not found! The app is falling back to a professional solid green gradient theme. Please place your stadium.jpg file inside the 'static/' folder to display it.")

# Check if model is trained, else display instructions
if model is None:
    st.error("❌ No trained model found in the `model/` directory! Please place your datasets in `dataset/` and run the training scripts first.")
    st.info("To train the model, open your terminal and run:\n1. `python preprocess.py`\n2. `python train_model.py`")
else:
    # Define active teams and common cities
    teams = [
        'Chennai Super Kings', 'Delhi Capitals', 'Gujarat Titans',
        'Kolkata Knight Riders', 'Lucknow Super Giants', 'Mumbai Indians',
        'Punjab Kings', 'Rajasthan Royals', 'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]
    
    cities = [
        'Mumbai', 'Kolkata', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
        'Jaipur', 'Ahmedabad', 'Lucknow', 'Chandigarh', 'Pune', 'Abu Dhabi', 
        'Dubai', 'Sharjah', 'Dharamshala', 'Visakhapatnam'
    ]

    # Main Application Layout: Responsive columns for laptop and desktop screens
    col1, col2 = st.columns([3, 2], gap="large")

    # Column 1: Match Settings Inputs
    with col1:
        st.markdown('<div class="card"><div class="card-title">🏟️ Match Setup</div>', unsafe_allow_html=True)
        
        # 1. Team Selection
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            batting_team = st.selectbox("Batting Team (Chasing)", teams, index=5) # Default: Mumbai Indians
        with sub_col2:
            bowling_options = [t for t in teams if t != batting_team]
            bowling_team = st.selectbox("Bowling Team (Defending)", bowling_options, index=7) # Default: RCB
            
        # 2. City Selector
        city = st.selectbox("Host City (Venue)", sorted(cities), index=10) # Default: Mumbai
        
        # 3. Toss settings
        sub_col3, sub_col4 = st.columns(2)
        with sub_col3:
            toss_winner = st.selectbox("Toss Winner", [batting_team, bowling_team])
        with sub_col4:
            toss_decision = st.selectbox("Toss Decision", ["Bat First", "Field First"])
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card"><div class="card-title">📈 Match Situation (Second Innings)</div>', unsafe_allow_html=True)
        
        # 4. Score metrics
        sub_col5, sub_col6 = st.columns(2)
        with sub_col5:
            target = st.number_input("Target Score", min_value=1, max_value=300, value=180, step=1)
        with sub_col6:
            runs_left = st.number_input("Runs Needed (Runs Left)", min_value=1, max_value=300, value=60, step=1)
            
        # 5. Over & Wickets metrics
        sub_col7, sub_col8 = st.columns(2)
        with sub_col7:
            balls_left = st.slider("Balls Remaining (Balls Left)", min_value=1, max_value=120, value=30, step=1)
        with sub_col8:
            wickets_left = st.slider("Wickets in Hand (Wickets Left)", min_value=1, max_value=10, value=6, step=1)
            
        # 6. Run rates calculation (autofills based on inputs, but allows manual adjustments)
        balls_bowled = 120 - balls_left
        current_score = target - runs_left
        
        auto_crr = (current_score * 6) / balls_bowled if balls_bowled > 0 else 0.0
        auto_rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0.0
        
        sub_col9, sub_col10 = st.columns(2)
        with sub_col9:
            crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, max_value=100.0, value=float(np.round(auto_crr, 2)), step=0.05)
        with sub_col10:
            rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0, max_value=100.0, value=float(np.round(auto_rrr, 2)), step=0.05)
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict Button
        predict_btn = st.button("🔮 Predict Win Probability")

    # Column 2: Predictions & Probability Visuals
    with col2:
        if predict_btn:
            # 1. Package inputs into DataFrame
            input_df = pd.DataFrame({
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
            
            # 2. Get probabilities
            probabilities = model.predict_proba(input_df)[0]
            batting_prob = probabilities[1]
            bowling_prob = probabilities[0]
            
            # 3. Determine the predicted winner
            predicted_winner = batting_team if batting_prob > bowling_prob else bowling_team
            winner_prob = max(batting_prob, bowling_prob)
            
            # 4. Display results inside a premium glassmorphism result card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center; color: #FFD54F !important; font-size: 1.4rem; margin-bottom: 5px;'>Predicted Winner</h3>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 2.2rem; font-weight: 800; text-align: center; color: #ffffff; margin-bottom: 20px; text-shadow: 0 0 10px rgba(255,255,255,0.3);'>🏆 {predicted_winner}</div>", unsafe_allow_html=True)
            
            st.divider()
            
            # Display win probabilities with progress bars
            st.subheader(f"📊 Win Probability Breakdown")
            
            # Chasing Team Prob
            st.write(f"**{batting_team} (Chasing)**")
            st.progress(float(batting_prob))
            st.markdown(f"<div class='probability-display' style='background: linear-gradient(90deg, #FFD54F 0%, #ffe082 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{batting_prob * 100:.1f}%</div>", unsafe_allow_html=True)
            
            # Defending Team Prob
            st.write(f"**{bowling_team} (Defending)**")
            st.progress(float(bowling_prob))
            st.markdown(f"<div class='probability-display' style='background: linear-gradient(90deg, #1565c0 0%, #42a5f5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>{bowling_prob * 100:.1f}%</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 5. Donut chart display
            st.markdown('<div class="card"><div class="card-title" style="text-align: center;">📈 Probability Donut Chart</div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(6, 6))
            colors = ['#FFD54F', '#1565C0']
            wedges, texts, autotexts = ax.pie(
                [batting_prob * 100, bowling_prob * 100], 
                labels=[batting_team, bowling_team], 
                autopct='%1.1f%%',
                startangle=90, 
                colors=colors, 
                textprops=dict(color="w", weight="bold", size=12),
                wedgeprops=dict(width=0.4, edgecolor='none') # Donut hole
            )
            
            # Stylize label texts for visual clarity
            for t in texts:
                t.set_color('#ffffff')
                t.set_fontsize(11)
                t.set_weight('bold')
            for at in autotexts:
                at.set_color('#ffffff')
                at.set_fontsize(12)
                at.set_weight('bold')

            # Render details cleanly
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            plt.tight_layout()
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="result-card" style="text-align: center;">', unsafe_allow_html=True)
            st.info("Configure the match situation and click **Predict Win Probability** to see the magic!")
            st.markdown('</div>', unsafe_allow_html=True)

# Footer Section
st.markdown("""
<div class="footer">
    Developed by Spandana T L | BE Artificial Intelligence and Data Science
</div>
""", unsafe_allow_html=True)
