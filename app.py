import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load robot's brain
model = joblib.load("sentiment_model.pkl")

# Labels dictionary
labels = {0: "Angry 😡", 1: "Happy 😊", 2: "Neutral 😐"}

st.title("💬 Sentiment Analyzer Dashboard")
st.write("This AI robot can read comments and guess mood: Happy, Angry, or Neutral!")

# --- 1. Single Comment Prediction ---
st.subheader("🔹 Try a Single Comment")
user_input = st.text_area("Type a comment here:")
if st.button("Analyze"):
    if user_input.strip():
        mood = model.predict([user_input])[0]
        if mood == 1:
            st.success("😊 Positive")
        elif mood == 0:
            st.error("😡 Negative")
        else:
            st.info("😐 Neutral")
    else:
        st.warning("Please type something!")

# --- 2. Upload CSV for Bulk Analysis ---
st.subheader("🔹 Upload Comments (CSV)")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "text" in df.columns:
        df["prediction"] = model.predict(df["text"])

        # Show summary in colored boxes
        pos = (df["prediction"] == 1).sum()
        neg = (df["prediction"] == 0).sum()
        neu = (df["prediction"] == 2).sum()

        st.success(f"😊 Positive: {pos}")
        st.error(f"😡 Negative: {neg}")
        st.info(f"😐 Neutral: {neu}")

        # Show sample results
        st.subheader("📋 Sample Results")
        st.write(df.head())

        # --- Pie Chart ---
        st.subheader("📊 Sentiment Distribution")
        counts = df["prediction"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts, labels=[labels[i] for i in counts.index], autopct="%1.1f%%")
        st.pyplot(fig)

        # --- Bar Chart ---
        st.subheader("📊 Sentiment Count (Bar Chart)")
        st.bar_chart(df["prediction"].value_counts())



        # --- Word Cloud ---
        st.subheader("☁️ Word Cloud (All Comments)")
        text_all = " ".join(df["text"].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_all)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

        # --- Download Results ---
        st.subheader("⬇️ Download Results")
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False),
            file_name="analyzed_comments.csv",
            mime="text/csv"
        )
    else:
        st.error("CSV must have a column named 'text'")
