import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# 1. Load CSV (use small sample for speed)
df = pd.read_csv("comments_sample.csv")  # or "acl_imdb_all.csv" if you want the whole dataset

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=42
)

# 3. Build a simple pipeline: TF-IDF converts text to numbers, LogisticRegression learns
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=5000)),
    ("clf", LogisticRegression(max_iter=200))
])

# 4. Train the model
print("Training the model now (this may take a little while)...")
pipe.fit(X_train, y_train)

# 5. Test and show results
y_pred = pipe.predict(X_test)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# 6. Save the trained model so we can use it later
joblib.dump(pipe, "sentiment_model.joblib")
print("Saved model to sentiment_model.joblib")
