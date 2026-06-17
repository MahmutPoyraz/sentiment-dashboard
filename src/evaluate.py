import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and print metrics.
    Returns a dictionary of metrics.
    """
    print(f"\nEvaluating {model_name}...")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='Positive')
    recall = recall_score(y_test, y_pred, pos_label='Positive')
    f1 = f1_score(y_test, y_pred, pos_label='Positive')

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save confusion matrix plot
    save_confusion_matrix(y_test, y_pred, model_name)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def save_confusion_matrix(y_test, y_pred, model_name):
    """Generate and save confusion matrix as an image."""
    cm = confusion_matrix(y_test, y_pred, labels=['Positive', 'Negative'])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Positive', 'Negative'],
        yticklabels=['Positive', 'Negative']
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    filename = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    filepath = os.path.join(MODELS_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Confusion matrix saved to {filepath}")


def plot_model_comparison(results):
    """
    Plot a bar chart comparing all models across metrics.
    results: dict of {model_name: {accuracy, precision, recall, f1_score}}
    """
    model_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[m][metric] for m in model_names]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — All Metrics')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0.8, 1.0)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    filepath = os.path.join(MODELS_DIR, 'model_comparison.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Model comparison chart saved to {filepath}")