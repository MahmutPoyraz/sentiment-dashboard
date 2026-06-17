import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.ft.txt')
TEST_FILE = os.path.join(DATA_DIR, 'test.ft.txt')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

STOP_WORDS = set(stopwords.words('english'))


def load_data(filepath, max_samples=None):
    """
    Load FastText format data file.
    Each line: __label__1 or __label__2 followed by review text.
    """
    texts = []
    labels = []

    print(f"Loading data from {filepath}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            # Split label from text
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            label_raw, text = parts
            # __label__1 = Negative, __label__2 = Positive
            label = 'Positive' if label_raw == '__label__2' else 'Negative'
            texts.append(text)
            labels.append(label)

    print(f"Loaded {len(texts)} samples.")
    return pd.DataFrame({'text': texts, 'sentiment': labels})


def clean_text(text):
    """
    Clean and normalize a single text string.
    Steps: lowercase, remove HTML, remove punctuation, remove digits, strip whitespace.
    """
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    """Remove English stopwords from tokenized text."""
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in STOP_WORDS and len(word) > 1]
    return ' '.join(filtered)


def preprocess_text(text):
    """Full preprocessing pipeline for a single text."""
    text = clean_text(text)
    text = remove_stopwords(text)
    return text


def preprocess_dataframe(df):
    """Apply full preprocessing pipeline to a DataFrame."""
    print("Preprocessing texts...")
    df = df.copy()
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Preprocessing complete.")
    return df


def build_tfidf_features(train_texts, test_texts, max_features=50000):
    """
    Fit TF-IDF vectorizer on training data and transform both train and test sets.
    Saves the vectorizer to disk for later use in the dashboard.
    """
    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Save vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"TF-IDF vectorizer saved to {VECTORIZER_PATH}")

    return X_train, X_test, vectorizer


def load_and_prepare_data(train_samples=200000, test_samples=40000):
    """
    Full pipeline: load → preprocess → vectorize.
    Returns X_train, X_test, y_train, y_test.
    """
    # Load
    train_df = load_data(TRAIN_FILE, max_samples=train_samples)
    test_df = load_data(TEST_FILE, max_samples=test_samples)

    # Preprocess
    train_df = preprocess_dataframe(train_df)
    test_df = preprocess_dataframe(test_df)

    # Labels
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    # TF-IDF
    X_train, X_test, _ = build_tfidf_features(
        train_df['cleaned_text'],
        test_df['cleaned_text']
    )

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"\nSentiment distribution (train):\n{y_train.value_counts()}")