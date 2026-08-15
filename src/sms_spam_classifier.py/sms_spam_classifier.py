import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

sms = pd.read_csv(r"C:\Users\Rigved Bhondve\OneDrive\Desktop\ML\SMS Spam Classification\data\raw\sms_spam_collection.csv")
X = sms["message"]
y = sms["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

THRESHOLD = 0.19
pipeline.fit(X_train, y_train)

new_message = [
    ""
]

spam_probabilities = pipeline.predict_proba(new_message)[:, 1]

predictions = np.where(
    spam_probabilities >= THRESHOLD,
    "spam",
    "ham"
)

for message, probability, prediction in zip(
    new_message,
    spam_probabilities,
    predictions
    ):
    print(f"Message: {message}")
    print(f"Spam probability: {probability:.3f}")
    print(f"Prediction: {prediction}")
    print()

joblib.dump(pipeline, "SMS_Spam_Predictor.pkl")