import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Load dataset
data = pd.read_csv("health_news.csv")  # make sure this file exists in your folder

# 2. Split features and labels
X = data["text"]
y = data["label"]

# 3. Convert text to numbers
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 4. Train model
model = LogisticRegression()
model.fit(X_vectorized, y)

# 5. Save model and vectorizer separately
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Training complete. model.pkl and vectorizer.pkl saved.")
