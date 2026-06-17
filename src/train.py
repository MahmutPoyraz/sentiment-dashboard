import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from src.preprocess import load_and_prepare_data
from src.database import init_db, insert_model_metrics
from src.evaluate import evaluate_model

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def train_all_models():
    """Load data, train all models, evaluate and save them."""

    # Initialize database
    init_db()

    # Load and preprocess data
    print("=" * 60)
    print("Loading and preprocessing data...")
    print("=" * 60)
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver='saga',
            n_jobs=-1,
            random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'SVM': CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, random_state=42)
        )
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Training {model_name}...")
        print("=" * 60)

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, model_name)
        results[model_name] = metrics

        # Save metrics to database
        insert_model_metrics(
            model_name=model_name,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score']
        )

        # Save model to disk
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_filename = model_name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(MODELS_DIR, model_filename)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE — MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 65)
    for model_name, metrics in results.items():
        print(
            f"{model_name:<25} "
            f"{metrics['accuracy']:>10.4f} "
            f"{metrics['precision']:>10.4f} "
            f"{metrics['recall']:>10.4f} "
            f"{metrics['f1_score']:>10.4f}"
        )

    return results


if __name__ == '__main__':
    train_all_models()