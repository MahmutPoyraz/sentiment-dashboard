import sqlite3
import os
from datetime import datetime

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'sentiment.db')


def get_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            predicted_sentiment TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            model_used TEXT NOT NULL,
            classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def insert_review(review_text, predicted_sentiment, confidence_score, model_used):
    """Insert a classified review into the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO reviews (review_text, predicted_sentiment, confidence_score, model_used, classified_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (review_text, predicted_sentiment, confidence_score, model_used, datetime.now()))

    conn.commit()
    conn.close()


def fetch_all_reviews():
    """Fetch all reviews from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM reviews ORDER BY classified_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def fetch_sentiment_stats():
    """Fetch sentiment distribution statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT predicted_sentiment, COUNT(*) as count
        FROM reviews
        GROUP BY predicted_sentiment
    ''')
    rows = cursor.fetchall()
    conn.close()
    return {row['predicted_sentiment']: row['count'] for row in rows}


def insert_model_metrics(model_name, accuracy, precision, recall, f1_score):
    """Insert model evaluation metrics into the database."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO model_metrics (model_name, accuracy, precision, recall, f1_score)
        VALUES (?, ?, ?, ?, ?)
    ''', (model_name, accuracy, precision, recall, f1_score))

    conn.commit()
    conn.close()


def fetch_model_metrics():
    """Fetch all model metrics from the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM model_metrics ORDER BY f1_score DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


if __name__ == '__main__':
    init_db()