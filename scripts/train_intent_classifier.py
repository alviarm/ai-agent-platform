#!/usr/bin/env python3
"""Script to train the intent classification model."""

import sys
sys.path.insert(0, "src")

from src.intent_classifier.train import train_model
from src.intent_classifier.evaluate import evaluate_model
from src.intent_classifier.export_onnx import export_to_onnx

if __name__ == "__main__":
    print("=" * 60)
    print("Training Intent Classification Model")
    print("=" * 60)
    
    # Train model
    model_path = train_model()
    
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    
    # Evaluate
    evaluate_model(
        model_path=model_path,
        output_dir="data/models/evaluation",
    )
    
    print("\n" + "=" * 60)
    print("Exporting to ONNX")
    print("=" * 60)
    
    # Export to ONNX
    export_to_onnx(
        model_path=model_path,
        output_path="data/models/onnx",
        benchmark=True,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"PyTorch model: {model_path}")
    print(f"ONNX model: data/models/onnx")
    print(f"Evaluation results: data/models/evaluation")
