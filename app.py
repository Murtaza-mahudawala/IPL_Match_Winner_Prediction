import streamlit as st
import pickle
import pandas as pd

# Teams and cities lists
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Set the title of the app
st.title('IPL Win Predictor')

# Create columns for layout
col1, col2 = st.columns(2)

# Team selection
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

# City selection
selected_city = st.selectbox('Select host city', sorted(cities))

# Target input
target = st.number_input('Target', min_value=1)

# Columns for score, overs, and wickets
col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, format="%.1f")  # Allows decimal input
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=9)  # Max wickets can be 9

# Prediction button
if st.button('Predict Probability'):
    # Calculate runs and balls left
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    
    # Handle cases where calculations might go negative
    if balls_left <= 0:
        st.error("Overs completed must be less than the total overs.")
    elif runs_left < 0:
        st.error("Score cannot be greater than or equal to target.")
    else:
        # Calculate the required run rate and current run rate
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0  # Avoid division by zero
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')  # Avoid division by zero

        # Create input DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Make prediction
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        
        # Display the results
        st.header(f"{batting_team} - {round(win * 100)}%")
        st.header(f"{bowling_team} - {round(loss * 100)}%")
