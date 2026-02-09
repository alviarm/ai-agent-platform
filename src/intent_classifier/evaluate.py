"""Evaluation script for intent classification model."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from .config import INTENT_CATEGORIES, MODEL_DIR, RANDOM_SEED
from .data_loader import create_datasets
from .model import IntentClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    output_dir: str = None,
    use_onnx: bool = False,
) -> Dict:
    """Evaluate the intent classification model.
    
    Args:
        model_path: Path to the model
        output_dir: Directory to save evaluation results
        use_onnx: Whether to use ONNX model
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model and data
    logger.info(f"Loading model from {model_path}...")
    classifier = IntentClassifier(model_path=model_path, use_onnx=use_onnx)
    
    logger.info("Loading test dataset...")
    datasets = create_datasets()
    test_dataset = datasets["test"]
    
    # Get predictions
    logger.info("Generating predictions...")
    texts = test_dataset["text"]
    true_labels = test_dataset["label"]
    
    predictions = []
    confidences = []
    
    # Batch predictions for efficiency
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i + batch_size]
        results = classifier.predict(batch_texts)
        for result in results:
            predictions.append(classifier.label2id[result["intent"]])
            confidences.append(result["confidence"])
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=0)
    
    # Per-class metrics
    class_report = classification_report(
        true_labels,
        predictions,
        target_names=INTENT_CATEGORIES,
        output_dict=True,
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    results = {
        "overall": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        },
        "per_class": class_report,
        "confusion_matrix": cm.tolist(),
        "average_confidence": float(np.mean(confidences)),
    }
    
    # Log results
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"Avg Confidence: {np.mean(confidences):.4f}")
    logger.info(f"\nClassification Report:\n{classification_report(true_labels, predictions, target_names=INTENT_CATEGORIES)}")
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        with open(f"{output_dir}/evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(cm, INTENT_CATEGORIES, output_dir)
        
        logger.info(f"\nResults saved to {output_dir}")
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    output_dir: str,
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix - Intent Classification")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    
    # Also create normalized version
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Normalized Confusion Matrix - Intent Classification")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_normalized.png", dpi=150)
    plt.close()


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate intent classification model")
    parser.add_argument("--model-path", default=f"{MODEL_DIR}/pytorch_model", help="Path to model")
    parser.add_argument("--output-dir", default=f"{MODEL_DIR}/evaluation", help="Output directory")
    parser.add_argument("--use-onnx", action="store_true", help="Use ONNX model")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_onnx=args.use_onnx,
    )


if __name__ == "__main__":
    main()
