"""Data loading and preprocessing for intent classification."""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .config import (
    INTENT_CATEGORIES,
    RANDOM_SEED,
    SYNTHETIC_DATA_SIZE,
    TEST_SIZE,
)

# Synthetic training templates for each intent
TEMPLATES: Dict[str, List[str]] = {
    "return": [
        "I want to return my order",
        "How do I return this product?",
        "This item is defective, I need to return it",
        "Can I get a refund for my purchase?",
        "The product doesn't match the description, returning it",
        "I received the wrong item, need to return",
        "Return policy for online purchases?",
        "My order arrived damaged, need return label",
        "How long do I have to return this?",
        "Initiate return for order #{order_id}",
        "This doesn't fit, I want to return it",
        "Changed my mind, can I return?",
        "Return authorization needed",
        "I want my money back",
        "Product quality is poor, returning",
    ],
    "grievance": [
        "I'm very disappointed with your service",
        "This is unacceptable, I want to speak to a manager",
        "Your company has terrible customer service",
        "I've been waiting for weeks with no response",
        "This is the worst experience I've ever had",
        "I want to file a complaint",
        "Your product caused damage to my property",
        "I'm extremely frustrated with this situation",
        "Nobody is helping me with my problem",
        "I demand compensation for this inconvenience",
        "This is false advertising",
        "Your staff was rude and unhelpful",
        "I'm considering legal action",
        "This company doesn't care about customers",
        "Multiple issues and no resolution",
    ],
    "billing": [
        "I was charged twice for this order",
        "There's an error on my invoice",
        "Why was I charged extra fees?",
        "I need to update my payment method",
        "My refund hasn't appeared yet",
        "Unauthorized charge on my account",
        "Billing question about my subscription",
        "Price discrepancy on my receipt",
        "Need a copy of my invoice",
        "When will I be charged for this?",
        "Remove the late fee from my account",
        "Payment failed but money was deducted",
        "Apply my promo code to this order",
        "Why is my bill higher than expected?",
        "Set up automatic payments",
    ],
    "technical": [
        "The app keeps crashing when I try to login",
        "I can't access my account",
        "Website is not loading properly",
        "Getting error code 500",
        "My password reset isn't working",
        "Two-factor authentication not sending codes",
        "API integration is failing",
        "System timeout during checkout",
        "Can't download my digital purchase",
        "Login credentials not recognized",
        "App freezes on iPhone 14",
        "Website buttons not responding",
        "Session keeps expiring",
        "Payment gateway error",
        "Images not loading on product pages",
    ],
    "support": [
        "How do I set up my new device?",
        "User manual for model XYZ?",
        "What are the product specifications?",
        "How do I clean this product?",
        "Compatible accessories for this item?",
        "Warranty information needed",
        "How to troubleshoot common issues?",
        "Software update instructions",
        "Installation guide request",
        "Product comparison information",
        "Size guide for clothing",
        "Compatibility with other devices",
        "Best practices for maintenance",
        "Feature explanation needed",
        "How to contact the manufacturer?",
    ],
    "general_inquiry": [
        "What are your store hours?",
        "Do you ship internationally?",
        "How long does shipping take?",
        "What payment methods do you accept?",
        "Do you have this in stock?",
        "Can I track my order?",
        "Where is my package?",
        "Gift card balance check",
        "Are there any current promotions?",
        "How do I create an account?",
        "Cancel my subscription",
        "Change my email address",
        "Unsubscribe from emails",
        "Loyalty program details",
        "Job openings at your company",
    ],
    "escalation": [
        "This is urgent, I need immediate help",
        "I've contacted you 5 times already",
        "Connect me to a supervisor right now",
        "This issue needs priority handling",
        "I'm a premium customer, treat me accordingly",
        "Escalate this to the technical team",
        "No first-level support, need senior engineer",
        "Business critical issue, need resolution today",
        "Contract requires 24h response time",
        "Executive escalation needed",
        "SLA breach imminent",
        "Production system down",
        "Revenue impact, urgent attention required",
        "Legal team should be involved",
        "Media inquiry about this issue",
    ],
}


def generate_synthetic_data(
    n_samples: int = SYNTHETIC_DATA_SIZE,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[int]]:
    """Generate synthetic training data from templates."""
    random.seed(seed)
    np.random.seed(seed)
    
    texts = []
    labels = []
    
    samples_per_intent = n_samples // len(INTENT_CATEGORIES)
    
    for intent_idx, intent in enumerate(INTENT_CATEGORIES):
        templates = TEMPLATES[intent]
        for _ in range(samples_per_intent):
            template = random.choice(templates)
            # Add some variations
            text = template.format(order_id=random.randint(10000, 99999))
            texts.append(text)
            labels.append(intent_idx)
    
    return texts, labels


def create_datasets(
    texts: List[str] = None,
    labels: List[int] = None,
    test_size: float = TEST_SIZE,
) -> DatasetDict:
    """Create train/validation/test datasets."""
    if texts is None:
        texts, labels = generate_synthetic_data()
    
    # Split into train+val and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
    )
    
    # Split train+val into train and val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.2, 
        random_state=RANDOM_SEED, stratify=train_val_labels
    )
    
    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "validation": Dataset.from_dict({"text": val_texts, "label": val_labels}),
        "test": Dataset.from_dict({"text": test_texts, "label": test_labels}),
    })
    
    return dataset


def get_label_distribution(dataset: DatasetDict) -> Dict[str, Dict[str, int]]:
    """Get distribution of labels in each split."""
    distribution = {}
    for split in dataset.keys():
        labels = dataset[split]["label"]
        counts = {}
        for label in labels:
            intent = INTENT_CATEGORIES[label]
            counts[intent] = counts.get(intent, 0) + 1
        distribution[split] = counts
    return distribution
