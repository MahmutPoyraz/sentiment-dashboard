# Lexara — Sentiment Analysis Dashboard

An end-to-end NLP sentiment analysis system that classifies product reviews as **Positive** or **Negative** using multiple machine learning models, stores results in a SQL database, and visualizes trends through an interactive dashboard.

---

## System Design

```
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                         │
│   Amazon Reviews Dataset (train.ft.txt / test.ft.txt)   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                 Preprocessing Pipeline                  │
│   Cleaning → Lowercasing → Stopword Removal →           │
│   Tokenization → TF-IDF Feature Extraction              │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  ML Model Training                      │
│   Logistic Regression │ Naive Bayes │ SVM (LinearSVC)   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                  Evaluation & Storage                   │
│   Accuracy, Precision, Recall, F1, Confusion Matrix     │
│   Results stored in SQLite via SQLAlchemy               │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│               Streamlit Dashboard (Lexara)              │
│ Analyze Review │ Statistics │ Model Comparison │ History│
└─────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [Amazon Reviews — Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)

The dataset contains real Amazon product reviews in FastText format. Each line begins with a sentiment label followed by the review text:

```
__label__2 This product is amazing and works perfectly...
__label__1 Terrible quality, broke after one week...
```

| Label | Sentiment | Count (train) |
|-------|-----------|---------------|
| `__label__2` | Positive | 101,166 |
| `__label__1` | Negative | 98,834 |

**Training samples used:** 200,000  
**Test samples used:** 40,000  
**Class balance:** Near-perfect (50.6% / 49.4%)

> Note: The dataset contains binary sentiment labels only. Neutral classification was not applicable as the source data does not include neutral labels.

---

## Model Selection Rationale

Three models were selected to cover a range of complexity and interpretability:

**Logistic Regression**  
Fast, interpretable, and highly effective on high-dimensional sparse TF-IDF features. Known to perform well on text classification tasks. Selected as the default model due to its best overall F1-Score.

**Naive Bayes (Multinomial)**  
A probabilistic baseline well-suited for text classification. Efficient training and naturally handles word count features. Included as a lightweight comparison point.

**Support Vector Machine (LinearSVC + Calibration)**  
A strong performer on text data, particularly with high-dimensional feature spaces. `CalibratedClassifierCV` was used to enable probability estimates required for confidence scoring.

---

## Training Process

1. Load 200,000 training samples from `train.ft.txt`
2. Apply full NLP preprocessing pipeline
3. Extract TF-IDF features (max 50,000 features, unigrams + bigrams)
4. Train each model on the vectorized training set
5. Evaluate on 40,000 test samples
6. Save models and vectorizer as `.pkl` files
7. Store evaluation metrics in SQLite database

To retrain all models:

```bash
python -m src.train
```

---

## Evaluation Results

| Model |                 Accuracy | Precision | Recall | F1-Score |
|-------|                |---------|-----------|--------|----------|
| Logistic Regression |  | 90.15%  |   89.92%  | 90.83% |  90.37%  |
| SVM |                  | 90.10%  |   89.93%  | 90.69% |  90.31%  |
| Naive Bayes |          | 87.71%  |   88.07%  | 87.74% |  87.90%  |

**Best model:** Logistic Regression (highest F1-Score)

Confusion matrices and model comparison charts are saved to the `models/` directory and are visible inside the dashboard under the **Model Comparison** tab.

---

## Project Structure

```
sentiment-dashboard/
├── data/
│   ├── train.ft.txt        # Training data (not tracked by git)
│   └── test.ft.txt         # Test data (not tracked by git)
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── *.png               # Confusion matrix images
├── src/
│   ├── preprocess.py       # NLP pipeline + TF-IDF
│   ├── train.py            # Model training + saving
│   ├── evaluate.py         # Metrics + confusion matrix
│   └── database.py         # SQLite schema + queries
├── dashboard/
│   └── app.py              # Streamlit dashboard
├── notebooks/
│   └── exploration.ipynb   # Exploratory analysis
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-dashboard.git
cd sentiment-dashboard
```

### 2. Create virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews) and place the files in the `data/` directory:

```
data/
├── train.ft.txt
└── test.ft.txt
```

### 5. Train the models

```bash
python -m src.train
```

This will:
- Preprocess the data
- Train all three models
- Save models to `models/`
- Store evaluation metrics in `sentiment.db`

### 6. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open your browser at `http://localhost:8501`

---

## Dashboard Overview

### Analyze Review
Submit any product review and receive an instant sentiment prediction with confidence score. Choose between Logistic Regression, Naive Bayes, or SVM.

### Statistics & Trends
View sentiment distribution, classification timeline, top words per sentiment category, and a word cloud visualization. All based on reviews submitted through the dashboard.

### Model Comparison
Side-by-side comparison of all three models across Accuracy, Precision, Recall, and F1-Score. Includes confusion matrices and a capability radar chart.

### Review History
Browse all previously analyzed reviews. Filter by sentiment or model. Includes confidence scores and timestamps.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| NLP | NLTK, scikit-learn |
| ML Models | scikit-learn |
| Feature Extraction | TF-IDF (50k features, bigrams) |
| Database | SQLite + SQLAlchemy |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, WordCloud |
| Version Control | Git / GitHub |

---

## Git Workflow

All development was tracked with meaningful commits following the convention:

- `feat:` — new feature or module
- `fix:` — bug fix
- `chore:` — configuration or dependency change
- `docs:` — documentation update