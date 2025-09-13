import joblib

# Load robot's brain
model = joblib.load("sentiment_model.pkl")

# Some test comments
samples = [
    "I love this new scheme, very helpful!",
    "This policy is useless and a waste of time.",
    "The meeting will be held tomorrow."
]

# Robot predicts mood
for text in samples:
    mood = model.predict([text])[0]
    label = {0: "Angry 😡", 1: "Happy 😊", 2: "Neutral 😐"}[mood]
    print(f"{text} → {label}")
