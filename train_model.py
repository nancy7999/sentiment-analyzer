import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load 3-class dataset
df = pd.read_csv("comments_3class.csv")

X = df["text"]
y = df["sentiment"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=300))
])

# Train the robot
model.fit(X_train, y_train)

# Test accuracy
print("🤖 Robot accuracy:", model.score(X_test, y_test))

# Save robot’s brain
joblib.dump(model, "sentiment_model.pkl")
print("✅ Saved robot’s brain as sentiment_model.pkl")
