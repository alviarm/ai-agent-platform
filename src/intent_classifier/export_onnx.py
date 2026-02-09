"""Export trained model to ONNX format for optimized inference."""

import argparse
import logging
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_LENGTH, MODEL_DIR, MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str = None,
    optimize: bool = True,
) -> str:
    """Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model directory
        output_path: Path for ONNX output
        optimize: Whether to optimize the ONNX model
        
    Returns:
        Path to the exported ONNX model
    """
    if output_path is None:
        output_path = f"{MODEL_DIR}/onnx"
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading model from {model_path}...")
    
    # Export using Optimum
    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            export=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save ONNX model
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"ONNX model exported to {output_path}")
        
        # Verify the export
        onnx_model_path = Path(output_path) / "model.onnx"
        if onnx_model_path.exists():
            logger.info(f"Model file: {onnx_model_path}")
            logger.info(f"File size: {onnx_model_path.stat().st_size / (1024*1024):.2f} MB")
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Optimum export failed: {e}")
        logger.info("Falling back to manual ONNX export...")
        return _export_manual(model_path, output_path)


def _export_manual(model_path: str, output_path: str) -> str:
    """Manual ONNX export as fallback."""
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.eval()
    
    # Create dummy input
    dummy_text = "This is a test input for ONNX export"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
    )
    
    # Export
    onnx_file = Path(output_path) / "model.onnx"
    
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_file,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"Manual ONNX export complete: {onnx_file}")
    
    return str(output_path)


def benchmark_onnx(
    onnx_path: str,
    pytorch_path: str,
    num_samples: int = 100,
) -> dict:
    """Benchmark ONNX vs PyTorch inference speed."""
    import time
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    # Test samples
    test_texts = [
        "I want to return my order",
        "This is urgent, help me now",
        "How do I reset my password?",
    ] * (num_samples // 3)
    
    logger.info("Benchmarking ONNX inference...")
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)
    session = ort.InferenceSession(f"{onnx_path}/model.onnx")
    
    # Warmup
    inputs = tokenizer(test_texts[0], return_tensors="np", padding=True, truncation=True)
    session.run(None, {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    })
    
    onnx_times = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
        start = time.time()
        session.run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        })
        onnx_times.append(time.time() - start)
    
    logger.info("Benchmarking PyTorch inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pt_model = AutoModelForSequenceClassification.from_pretrained(pytorch_path).to(device)
    pt_model.eval()
    pt_tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
    
    # Warmup
    inputs = pt_tokenizer(test_texts[0], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        pt_model(**inputs)
    
    pytorch_times = []
    for text in test_texts:
        inputs = pt_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        start = time.time()
        with torch.no_grad():
            pt_model(**inputs)
        pytorch_times.append(time.time() - start)
    
    results = {
        "onnx_avg_ms": np.mean(onnx_times) * 1000,
        "onnx_std_ms": np.std(onnx_times) * 1000,
        "pytorch_avg_ms": np.mean(pytorch_times) * 1000,
        "pytorch_std_ms": np.std(pytorch_times) * 1000,
        "speedup": np.mean(pytorch_times) / np.mean(onnx_times),
    }
    
    logger.info(f"\nBenchmark Results:")
    logger.info(f"  ONNX avg:     {results['onnx_avg_ms']:.2f} ± {results['onnx_std_ms']:.2f} ms")
    logger.info(f"  PyTorch avg:  {results['pytorch_avg_ms']:.2f} ± {results['pytorch_std_ms']:.2f} ms")
    logger.info(f"  Speedup:      {results['speedup']:.2f}x")
    
    return results


def main():
    """CLI entry point for ONNX export."""
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-path", default=f"{MODEL_DIR}/pytorch_model", help="Path to PyTorch model")
    parser.add_argument("--output-path", default=f"{MODEL_DIR}/onnx", help="Output path for ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark ONNX vs PyTorch")
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
    )
    
    if args.benchmark:
        benchmark_onnx(args.output_path, args.model_path)


if __name__ == "__main__":
    main()
