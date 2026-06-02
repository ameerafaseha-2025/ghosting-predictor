import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
import pickle


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Ghost Buster",
    page_icon="💔",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model("best_automl_ghosting_pipeline")

pca = pickle.load(open("pca_transformer.pkl","rb"))
features = pickle.load(open("feature_names.pkl","rb"))
pipeline = pickle.load(open("best_automl_ghosting_pipeline.pkl","rb"))


# ==========================================
# HEADER
# ==========================================

st.title("💔 Ghost Buster")

st.markdown(
"""
Predict the likelihood of ghosting based on dating app behaviour patterns.

Fill in the information below and press **Predict**.
"""
)

st.markdown("---")


# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Non-binary",
            "Transgender",
            "Genderfluid",
            "Prefer Not to Say"
        ]
    )

    location_type = st.selectbox(
        "Location Type",
        [
            "Urban",
            "Suburban",
            "Rural"
        ]
    )

    sexual_orientation = st.selectbox(
        "Sexual Orientation",
        [
            "Straight",
            "Gay",
            "Lesbian",
            "Bisexual",
            "Pansexual",
            "Asexual"
        ]
    )

    education_level = st.selectbox(
        "Education Level",
        [
            "No Formal Education",
            "Bachelor’s",
            "Master’s",
            "Postdoc"
        ]
    )

    income_bracket = st.selectbox(
        "Income Bracket",
        [
            "Very Low",
            "Low",
            "Middle",
            "Upper-Middle",
            "High",
            "Very High"
        ]
    )

    swipe_time_of_day = st.selectbox(
        "Swipe Time",
        [
            "Morning",
            "Afternoon",
            "Evening",
            "Night"
        ]
    )

with col2:

    app_usage_time_min = st.number_input(
        "App Usage Time (minutes)",
        0,
        1440,
        60
    )

    swipe_right_ratio = st.slider(
        "Swipe Right Ratio",
        0.0,
        1.0,
        0.5
    )

    likes_received = st.number_input(
        "Likes Received",
        0,
        10000,
        50
    )

    mutual_matches = st.number_input(
        "Mutual Matches",
        0,
        10000,
        20
    )

    profile_pics_count = st.slider(
        "Profile Pictures Count",
        0,
        20,
        4
    )

    bio_length = st.number_input(
        "Bio Length",
        0,
        1000,
        120
    )

    message_sent_count = st.number_input(
        "Messages Sent",
        0,
        10000,
        25
    )

    emoji_usage_rate = st.slider(
        "Emoji Usage Rate",
        0.0,
        1.0,
        0.3
    )

    last_active_hour = st.slider(
        "Last Active Hour",
        0,
        23,
        20
    )


st.markdown("---")


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({

    "gender":[gender],

    "sexual_orientation":[sexual_orientation],

    "location_type":[location_type],

    "income_bracket":[income_bracket],
    
    "education_level":[education_level],

    "app_usage_time_min":[app_usage_time_min],

    "swipe_right_ratio":[swipe_right_ratio],

    "likes_received":[likes_received],

    "mutual_matches":[mutual_matches],

    "profile_pics_count":[profile_pics_count],

    "bio_length":[bio_length],

    "message_sent_count":[message_sent_count],

    "emoji_usage_rate":[emoji_usage_rate],

    "last_active_hour":[last_active_hour],

    "swipe_time_of_day":[swipe_time_of_day]

})

income_mapping = {
    'Very Low': 0,
    'Low': 1,
    'Middle': 2,
    'Upper-Middle': 3,
    'High': 4,
    'Very High': 5
}

input_data["income_bracket"] = (
    input_data["income_bracket"]
    .map(income_mapping)
)

edu_map = {
    'No Formal Education': 0,
    'Bachelor’s': 1,
    'Master’s': 2,
    'Postdoc': 3
}

input_data["education_level"] = (
    input_data["education_level"]
    .map(edu_map)
)



if st.button(
    "Predict Ghosting Likelihood",
    use_container_width=True
):

    result = predict_model(
        model,
        data=input_data,
        raw_score=True
    )

    prediction = result["prediction_label"][0]

    probability = result[
        "prediction_score_1"
    ][0]


    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ High Likelihood of Ghosting"
        )

    else:

        st.success(
            "✅ Low Likelihood of Ghosting"
        )


    st.metric(
        "Ghosting Probability",
        f"{probability:.2%}"
    )

    st.progress(
        float(probability)
    )


st.markdown("---")

st.caption(
"""
WIA1006 Machine Learning

Ghost Buster Team
"""
)