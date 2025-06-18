# 2_train_model.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Import custom modules
from src.data_preparation import load_and_prepare_data
from src.preprocessing import preprocess_text

# Define constants
RAW_DATA_FILE = 'data/raw/Reviews.csv'
TRANSFORMER_PATH = 'models/transformer.pkl'
MODEL_PATH = 'models/model.pkl'

# --- 1. Load and Prepare Data ---
df = load_and_prepare_data(RAW_DATA_FILE)

# --- 2. Preprocess Text Data ---
print("Applying text preprocessing to all reviews. This may take a moment...")
df['processed_text'] = df['Text'].apply(preprocess_text)
print("Text preprocessing complete.")

# --- 3. Train-Test Split ---
X = df.processed_text
y = df.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Data split into training and testing sets: {X_train.shape[0]} train, {X_test.shape[0]} test")

# --- 4. Vectorization (TF-IDF) ---
print("Vectorizing text data using TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000)
tfidf_vectorizer.fit(X_train)

X_train_tfidf = tfidf_vectorizer.transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("Vectorization complete.")

# --- 5. Model Training ---
print("Training Logistic Regression model...")
# Based on your notebook's findings, C=1 was a good hyperparameter
model = LogisticRegression(C=1, max_iter=500, random_state=42)
model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- 6. Model Evaluation ---
print("Evaluating model performance...")
y_preds_test = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_preds_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plotting Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_preds_test, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, cmap='Blues', cbar=False, fmt='.2f',
    xticklabels=model.classes_, yticklabels=model.classes_
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig('models/confusion_matrix.png') # Save the plot
plt.show()

# --- 7. Save Model and Transformer ---
print(f"Saving TF-IDF vectorizer to {TRANSFORMER_PATH}")
with open(TRANSFORMER_PATH, "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print(f"Saving trained model to {MODEL_PATH}")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\nTraining pipeline finished successfully!")