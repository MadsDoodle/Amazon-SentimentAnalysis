# src/preprocessing.py

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- Stopwords Configuration ---
total_stopwords = set(stopwords.words('english'))
negative_stop_words = set(word for word in total_stopwords if "n't" in word or 'no' in word)
final_stopwords = total_stopwords - negative_stop_words
final_stopwords.add("one") # Add custom stopwords if needed

# --- Preprocessing Tools ---
stemmer = PorterStemmer()
HTMLTAGS = re.compile('<.*?>')
PUNCTUATION_TABLE = str.maketrans(dict.fromkeys(string.punctuation))
DIGITS_TABLE = str.maketrans('', '', string.digits)
MULTIPLE_WHITESPACE = re.compile(r"\s+")

def preprocess_text(review: str) -> str:
    """
    Cleans and preprocesses a single review text.
    """
    # remove html tags
    review = HTMLTAGS.sub(r'', review)
    # remove puncutuation
    review = review.translate(PUNCTUATION_TABLE)
    # remove digits
    review = review.translate(DIGITS_TABLE)
    # lower case all letters
    review = review.lower()
    # replace multiple white spaces with single space
    review = MULTIPLE_WHITESPACE.sub(" ", review).strip()
    # remove stop words
    review_words = [word for word in review.split() if word not in final_stopwords]
    # stemming
    stemmed_words = [stemmer.stem(word) for word in review_words]
    
    return ' '.join(stemmed_words)