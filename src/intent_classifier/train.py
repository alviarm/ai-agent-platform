"""Training script for intent classification model."""

import argparse
import logging
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import (
    BATCH_SIZE,
    INTENT_CATEGORIES,
    LEARNING_RATE,
    MAX_LENGTH,
    MODEL_DIR,
    MODEL_NAME,
    NUM_EPOCHS,
    RANDOM_SEED,
    WARMUP_RATIO,
)
from .data_loader import create_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def tokenize_function(examples, tokenizer):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def train_model(
    output_dir: str = MODEL_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    use_gpu: bool = True,
) -> str:
    """Train the intent classification model.
    
    Args:
        output_dir: Directory to save the model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        use_gpu: Whether to use GPU if available
        
    Returns:
        Path to the saved model
    """
    # Setup
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and create datasets
    logger.info("Loading tokenizer and creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    datasets = create_datasets()
    
    # Tokenize datasets
    tokenized_datasets = datasets.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
    )
    
    # Model setup
    logger.info("Initializing model...")
    label2id = {label: i for i, label in enumerate(INTENT_CATEGORIES)}
    id2label = {i: label for i, label in enumerate(INTENT_CATEGORIES)}
    
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENT_CATEGORIES),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=RANDOM_SEED,
        report_to="none",
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    logger.info(f"Test results: {test_results}")
    
    # Save model
    model_path = f"{output_dir}/pytorch_model"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train intent classification model")
    parser.add_argument("--output-dir", default=MODEL_DIR, help="Output directory")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    
    args = parser.parse_args()
    
    train_model(
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.no_gpu,
    )


if __name__ == "__main__":
    main()
