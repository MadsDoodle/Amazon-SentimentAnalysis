import streamlit as st
import os

# We will reuse the prediction logic from your existing script.
# This avoids code duplication and keeps concerns separated.
from predict_sentiment import get_sentiment 

# --- UI Configuration ---
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# --- Main App UI ---
st.title("Amazon Fine Food Review Sentiment Analysis 🛍️")

st.markdown("""
Welcome! Enter a review for an Amazon food product below, and the model will predict
whether the sentiment is **Positive**, **Negative**, or **Neutral**.
""")

# Check if the model files exist before showing the input form
MODEL_PATH = 'models/model.pkl'
TRANSFORMER_PATH = 'models/transformer.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(TRANSFORMER_PATH):
    st.error("Model files not found! 😥")
    st.info("Please run the training script `python 2_train_model.py` to generate the necessary model files.")
else:
    # --- Input Form ---
    with st.form("review_form"):
        review_text = st.text_area(
            "Enter your review text here:",
            height=150,
            placeholder="e.g., 'The product was absolutely fantastic, I would definitely buy it again!'"
        )
        submitted = st.form_submit_button("Analyze Sentiment")

    # --- Prediction Logic ---
    if submitted and review_text:
        with st.spinner('Analyzing...'):
            prediction = get_sentiment(review_text)
        
        st.write("---")
        st.subheader("Analysis Result")

        if prediction == "Positive":
            st.success(f"**Sentiment: Positive** 👍")
            st.balloons()
        elif prediction == "Negative":
            st.error(f"**Sentiment: Negative** 👎")
        else:
            st.warning(f"**Sentiment: Neutral** 😐")
