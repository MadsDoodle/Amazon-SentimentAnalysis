# 3_predict_sentiment.py

import pickle
import os

# Import the same preprocessing function used during training
from src.preprocessing import preprocess_text

# Define paths
TRANSFORMER_PATH = 'models/transformer.pkl'
MODEL_PATH = 'models/model.pkl'

def get_sentiment(review: str) -> str:
    """
    Loads the saved model and predicts the sentiment of a given review.
    
    Returns:
        Sentiment label ('Positive', 'Negative', 'Neutral') or an error message.
    """
    # Check if model files exist
    if not os.path.exists(TRANSFORMER_PATH) or not os.path.exists(MODEL_PATH):
        return "Error: Model or transformer not found. Please run 2_train_model.py first."

    # Load the saved vectorizer and model
    with open(TRANSFORMER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    # 1. Preprocess the input review
    processed_review = preprocess_text(review)
    
    # 2. Vectorize the preprocessed text
    vectorized_review = vectorizer.transform([processed_review])
    
    # 3. Predict the sentiment
    prediction = model.predict(vectorized_review)
    
    return prediction[0]

if __name__ == "__main__":
    # --- Example 1: Positive Review ---
    positive_review = "This chips packet is very tasty. I highly recommend this! The flavor is amazing."
    sentiment = get_sentiment(positive_review)
    print(f"Review: '{positive_review}'")
    print(f"Predicted Sentiment: {sentiment}\n")

    # --- Example 2: Negative Review ---
    negative_review = "This product is a complete waste of money. Don't buy this!! It tasted awful."
    sentiment = get_sentiment(negative_review)
    print(f"Review: '{negative_review}'")
    print(f"Predicted Sentiment: {sentiment}\n")

    # --- Example 3: Neutral Review ---
    neutral_review = "The delivery was on time but the packaging could have been better. The product itself is as described."
    sentiment = get_sentiment(neutral_review)
    print(f"Review: '{neutral_review}'")
    print(f"Predicted Sentiment: {sentiment}\n")