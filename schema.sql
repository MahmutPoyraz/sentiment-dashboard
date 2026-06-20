-- ============================================================
-- Sentiment Analysis Dashboard — Database Schema
-- ============================================================

-- Table: reviews
-- Stores all user-submitted reviews and their classification results
CREATE TABLE IF NOT EXISTS reviews (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    review_text         TEXT    NOT NULL,
    predicted_sentiment TEXT    NOT NULL,
    confidence_score    REAL    NOT NULL,
    model_used          TEXT    NOT NULL,
    classified_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: model_metrics
-- Stores evaluation metrics for each trained model
CREATE TABLE IF NOT EXISTS model_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name  TEXT NOT NULL,
    accuracy    REAL,
    precision   REAL,
    recall      REAL,
    f1_score    REAL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for faster querying
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment
    ON reviews (predicted_sentiment);

CREATE INDEX IF NOT EXISTS idx_reviews_model
    ON reviews (model_used);

CREATE INDEX IF NOT EXISTS idx_reviews_timestamp
    ON reviews (classified_at);