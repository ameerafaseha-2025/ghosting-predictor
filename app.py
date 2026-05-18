import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# TITLE
st.title("Likelihood of Ghosting Predictor")

st.write("Fill in the information below to predict the likelihood of ghosting.")

# LOAD DATA
data = pd.read_csv("ghosting_dataset.csv")

# FEATURES
X = data[["reply_time", "interest_level", "message_frequency", "relationship_stage"]]

# TARGET
y = data["ghosted"]

# TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

# USER INPUTS
reply_time = st.slider("Reply Time (hours)", 1, 24, 5)

interest_level = st.slider("Interest Level", 1, 10, 5)

message_frequency = st.slider("Messages per Day", 1, 20, 5)

relationship_stage = st.selectbox(
    "Relationship Stage",
    [1, 2, 3],
    format_func=lambda x:
    "Talking Stage" if x == 1 else
    "Close" if x == 2 else
    "Serious"
)

# PREDICT BUTTON
if st.button("Predict Ghosting Likelihood"):

    input_data = [[
        reply_time,
        interest_level,
        message_frequency,
        relationship_stage
    ]]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High chance of ghosting")
    else:
        st.success("Low chance of ghosting")

    st.write(f"Probability of ghosting: {probability:.2f}")