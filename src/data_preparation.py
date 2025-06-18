# src/data_preparation.py

import pandas as pd
import numpy as np

def create_sentiment_target(score: int) -> str:
    """Converts a numeric score (1-5) to a sentiment label."""
    if score > 3:
        return "Positive"
    elif score < 3:
        return "Negative"
    else:
        return "Neutral"

def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV, cleans it, balances it, and returns a prepared DataFrame.
    """
    print("Loading raw data...")
    data = pd.read_csv(file_path)

    # --- Initial Cleaning ---
    data.dropna(how='any', inplace=True)
    data.drop_duplicates(inplace=True, subset=['Score', 'Text'])
    
    # Remove illogical helpfulness scores
    invalid_helpfulness_idx = data[data["HelpfulnessNumerator"] > data["HelpfulnessDenominator"]].index
    data.drop(index=invalid_helpfulness_idx, inplace=True)

    print("Creating sentiment labels...")
    data['target'] = data['Score'].apply(create_sentiment_target)

    # --- Balancing the Dataset ---
    print("Balancing the dataset...")
    neutral = data[data.target == "Neutral"]
    positive = data[data.target == "Positive"].sample(n=50000, random_state=42)
    negative = data[data.target == "Negative"].sample(n=50000, random_state=42)

    balanced_data = pd.concat([positive, negative, neutral], ignore_index=True)
    print(f"Data prepared. Final shape: {balanced_data.shape}")
    
    return balanced_data