
import streamlit as st
import pandas as pd
import pickle

# PAGE CONFIG
st.set_page_config(
    page_title="Ghost Buster",
    page_icon="💔",
    layout="wide"
)

# LOAD MODEL
model = pickle.load(open("best_automl_ghosting_pipeline.pkl", "rb"))

# LOAD FEATURE NAMES
feature_names = pickle.load(open("feature_names.pkl", "rb"))

# TITLE
st.title("💔 Ghost Buster")
st.subheader("Predicting the Likelihood of Ghosting")

st.write("""
This machine learning dashboard predicts the likelihood of ghosting
based on dating app behavior patterns.
""")

# SIDEBAR
st.sidebar.header("User Behavior Input")

# USER INPUTS
income_bracket = st.sidebar.slider("Income Bracket", 0, 5, 2)

education_level = st.sidebar.slider("Education Level", 0, 3, 1)

app_usage_time_min = st.sidebar.slider(
    "Daily App Usage Time (minutes)",
    0,
    500,
    120
)

swipe_right_ratio = st.sidebar.slider(
    "Swipe Right Ratio",
    0,
    100,
    50
)

likes_received = st.sidebar.slider(
    "Likes Received",
    0,
    1000,
    100
)

mutual_matches = st.sidebar.slider(
    "Mutual Matches",
    0,
    500,
    50
)

message_sent_count = st.sidebar.slider(
    "Messages Sent",
    0,
    500,
    50
)

emoji_usage_rate = st.sidebar.slider(
    "Emoji Usage Rate",
    0.0,
    1.0,
    0.5
)

# INPUT DATA
input_dict = {
    'income_bracket': income_bracket,
    'education_level': education_level,
    'app_usage_time_min': app_usage_time_min,
    'swipe_right_ratio': swipe_right_ratio,
    'likes_received': likes_received,
    'mutual_matches': mutual_matches,
    'message_sent_count': message_sent_count,
    'emoji_usage_rate': emoji_usage_rate
}

# Create DataFrame
input_data = pd.DataFrame([input_dict])

# Add missing columns automatically
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder columns
input_data = input_data[feature_names]

# DISPLAY INPUT
st.subheader("📋 User Input")
st.write(input_data)

# PREDICT BUTTON
if st.button("Predict Ghosting Likelihood"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Likelihood of Ghosting")
    else:
        st.success("✅ Low Likelihood of Ghosting")

    st.metric(
        label="Ghosting Probability",
        value=f"{probability:.2%}"
    )

# FOOTER
st.markdown("---")

st.write("""
WIA1006/WID3006 Machine Learning Project  
Theme: Likelihood of Ghosting  
Group Project: Ghost Buster
""")