import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.DataFrame({
    "income_bracket":[1,2,3,4],
    "education_level":[1,2,2,3],
    "app_usage_time_min":[300,200,100,50],
    "swipe_right_ratio":[80,60,40,20],
    "likes_received":[500,300,100,20],
    "mutual_matches":[100,50,10,2],
    "message_sent_count":[200,150,50,10],
    "emoji_usage_rate":[0.9,0.7,0.3,0.1],
    "ghosted":[0,0,1,1]
})

X = data.drop("ghosted", axis=1)
y = data["ghosted"]

model = LogisticRegression()
model.fit(X, y)

# SAVE MODEL
pickle.dump(model, open("ghosting_model.pkl", "wb"))

# SAVE FEATURE NAMES
pickle.dump(list(X.columns), open("feature_names.pkl", "wb"))

print("Model and features saved successfully!")