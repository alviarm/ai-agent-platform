#!/usr/bin/env python3
"""
Synthetic Test Data Generator for Customer Service AI Platform

Generates:
- 200 labeled intent classification samples (balanced across categories)
- 50 conversation histories (multi-turn) for RAG testing
- Raw feedback data for analytics testing

Includes edge cases: ambiguous queries, typos, multilingual snippets
"""

import csv
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

# Ensure reproducibility
random.seed(42)

# Intent categories
INTENTS = ["return", "billing", "technical", "support", "general_inquiry", "escalation", "grievance"]

# Templates for each intent category
INTENT_TEMPLATES = {
    "return": [
        "I want to return my {product}",
        "How do I send back this {product}?",
        "This {product} is defective, need to return",
        "Return policy for {product}?",
        "I need a refund for my order",
        "The {product} arrived broken, want to return it",
        "Can I exchange this {product} for a different size?",
        "Initiating return for order #{order_id}",
        "My {product} doesn't fit, how to return?",
        "Return window for items bought last month?",
        "Got the wrong {product}, need to return",
        "How long do returns take to process?",
        "Return shipping label not received",
        "Can I return without the original box?",
        "Gift return question - recipient returning",
    ],
    "billing": [
        "I was charged twice for my order",
        "Why is there an extra charge on my card?",
        "My refund hasn't appeared yet",
        "Billing issue with order #{order_id}",
        "Unauthorized charge on my account",
        "Update my payment method",
        "Card declined but money was taken",
        "Price adjustment request",
        "Question about sales tax",
        "Double charged for shipping",
        "Payment not going through",
        "Need a copy of my invoice",
        "Pending charge question",
        "Afterpay installment issue",
        "Promo code didn't apply",
    ],
    "technical": [
        "App keeps crashing when I login",
        "Can't reset my password",
        "Two-factor authentication not working",
        "Website won't load properly",
        "Account login issues",
        "Error 500 on checkout",
        "Mobile app won't install",
        "Push notifications broken",
        "Session keeps expiring",
        "Can't update email address",
        "Saved payment methods disappeared",
        "Download link not working",
        "Account hacked, need help",
        "Unsubscribe from emails not working",
        "Browser compatibility issue",
    ],
    "support": [
        "How do I set up my {product}?",
        "Product specifications question",
        "Is this compatible with {device}?",
        "Warranty information",
        "How to contact support by phone?",
        "Looking for product reviews",
        "Bulk pricing available?",
        "Can I get a product demo?",
        "User manual request",
        "Product comparison help",
        "Accessory recommendations",
        "Upgrade options available?",
        "Product care instructions",
        "Warranty claim process",
        "Professional installation available?",
    ],
    "general_inquiry": [
        "What are your store hours?",
        "Do you ship to {country}?",
        "How long does shipping take?",
        "Where is my order?",
        "Order confirmation email missing",
        "Track my package",
        "Change shipping address",
        "Cancel my order",
        "Modify order before shipping",
        "Gift wrapping available?",
        "Corporate gifting options",
        "Newsletter subscription",
        "Career opportunities",
        "Partner with your company",
        "Press inquiry",
    ],
    "escalation": [
        "This is the third time I've contacted you",
        "I need to speak to a supervisor",
        "Escalate this issue immediately",
        "I've been waiting for a response for days",
        "Connect me to your manager",
        "This is unacceptable service",
        "I want to speak to corporate",
        "File a formal complaint",
        "Legal action will be taken",
        "Reporting to BBB",
        "Social media post coming",
        "Executive team contact needed",
        "Taking my business elsewhere",
        "Demand immediate callback",
        "Final warning before dispute",
    ],
    "grievance": [
        "Extremely dissatisfied with service",
        "Worst customer experience ever",
        "You ruined my {event}",
        "False advertising complaint",
        "Product caused damage",
        "Missed guaranteed delivery date",
        "Customer service was rude",
        "Misleading product description",
        "Want to cancel membership",
        "Never ordering again",
        "Regret this purchase",
        "Lost money because of you",
        "Emotional distress from experience",
        "Demand full compensation",
        "Reviewing my legal options",
    ],
}

# Products for templating
PRODUCTS = [
    "laptop", "phone", "headphones", "smartwatch", "tablet", "camera",
    "speaker", "keyboard", "mouse", "monitor", "charger", "cable",
    "case", "screen protector", "battery", "adapter", "router", "modem",
    "printer", "scanner", "microphone", "webcam", "USB drive", "hard drive",
]

# Devices for compatibility questions
DEVICES = [
    "iPhone 14", "Samsung Galaxy", "iPad Pro", "MacBook", "Windows PC",
    "Android tablet", "Chromebook", "Smart TV", "PlayStation 5", "Xbox",
]

# Countries for shipping questions
COUNTRIES = [
    "Canada", "UK", "Australia", "Germany", "France", "Japan",
    "Mexico", "Brazil", "India", "Singapore", "UAE", "Netherlands",
]

# Events for grievances
EVENTS = [
    "birthday", "wedding", "holiday", "anniversary", "business trip",
    "vacation", "graduation", "presentation", "interview",
]


def generate_order_id():
    """Generate realistic order ID."""
    return f"ORD-{random.randint(100000, 999999)}"


def apply_typos(text: str, typo_prob: float = 0.05) -> str:
    """Randomly introduce typos into text."""
    if random.random() > typo_prob:
        return text
    
    words = text.split()
    if len(words) < 2:
        return text
    
    typo_type = random.choice(["swap", "double", "missing", "replace"])
    idx = random.randint(0, len(words) - 1)
    word = words[idx]
    
    if len(word) < 3:
        return text
    
    if typo_type == "swap" and len(word) > 2:
        # Swap two adjacent characters
        char_idx = random.randint(0, len(word) - 2)
        word = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]
    elif typo_type == "double":
        # Double a character
        char_idx = random.randint(0, len(word) - 1)
        word = word[:char_idx+1] + word[char_idx] + word[char_idx+1:]
    elif typo_type == "missing" and len(word) > 3:
        # Remove a character
        char_idx = random.randint(0, len(word) - 1)
        word = word[:char_idx] + word[char_idx+1:]
    elif typo_type == "replace":
        # Replace a character with adjacent key
        keyboard_adjacent = {
            'a': 's', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh',
            'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k',
            'q': 'w', 'w': 'qe', 'e': 'wr', 'r': 'et', 't': 'ry',
            'y': 'tu', 'u': 'yi', 'i': 'uo', 'o': 'ip', 'p': 'o',
        }
        char_idx = random.randint(0, min(3, len(word) - 1))
        char = word[char_idx].lower()
        if char in keyboard_adjacent:
            replacement = random.choice(keyboard_adjacent[char])
            word = word[:char_idx] + replacement + word[char_idx+1:]
    
    words[idx] = word
    return " ".join(words)


def add_multilingual(text: str, prob: float = 0.03) -> str:
    """Occasionally add multilingual snippets."""
    if random.random() > prob:
        return text
    
    snippets = [
        " (gracias)",
        " - merci",
        " danke",
        " - 谢谢",
        " grazie",
        " obrigado",
    ]
    return text + random.choice(snippets)


def generate_intent_sample(intent: str, include_typos: bool = False) -> Dict:
    """Generate a single intent classification sample."""
    template = random.choice(INTENT_TEMPLATES[intent])
    
    # Fill in template
    text = template.format(
        product=random.choice(PRODUCTS),
        device=random.choice(DEVICES),
        country=random.choice(COUNTRIES),
        event=random.choice(EVENTS),
        order_id=generate_order_id(),
    )
    
    # Apply typos if requested
    if include_typos:
        text = apply_typos(text, typo_prob=0.15)
    
    # Occasionally add multilingual
    text = add_multilingual(text)
    
    return {
        "text": text,
        "intent": intent,
        "metadata": {
            "has_typos": include_typos,
            "template_used": template[:50] + "..." if len(template) > 50 else template,
        }
    }


def generate_ambiguous_query() -> Dict:
    """Generate an ambiguous query that could fit multiple intents."""
    ambiguous_templates = [
        "I want to return this but also question the charge",
        "My item is broken and I want a refund but also update my card",
        "Having issues with login and also want to cancel my order",
        "Product problem and billing error at the same time",
        "Can't access account to check my return status",
        "Wrong item sent and charged wrong amount",
        "App not working, need help with return",
        "Order never arrived but I was charged",
        "Return label not working, website broken",
        "Need to change address for pending order but can't login",
    ]
    
    text = random.choice(ambiguous_templates)
    # Label as escalation since it's complex
    return {
        "text": text,
        "intent": "escalation",
        "metadata": {
            "is_ambiguous": True,
            "possible_intents": ["return", "billing", "technical"],
        }
    }


def generate_intent_dataset(num_samples: int = 200) -> List[Dict]:
    """Generate balanced intent classification dataset."""
    print(f"Generating {num_samples} intent classification samples...")
    
    samples = []
    samples_per_intent = num_samples // len(INTENTS)
    
    # Generate balanced samples
    for intent in INTENTS:
        # 80% clean samples
        for _ in range(int(samples_per_intent * 0.8)):
            samples.append(generate_intent_sample(intent, include_typos=False))
        
        # 20% samples with typos
        for _ in range(int(samples_per_intent * 0.2)):
            samples.append(generate_intent_sample(intent, include_typos=True))
    
    # Add ambiguous queries (about 5% of total)
    num_ambiguous = num_samples // 20
    for _ in range(num_ambiguous):
        samples.append(generate_ambiguous_query())
    
    # Shuffle
    random.shuffle(samples)
    
    return samples[:num_samples]


def generate_conversation_turn(role: str, content: str, intent: str = None) -> Dict:
    """Generate a single conversation turn."""
    return {
        "role": role,
        "content": content,
        "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
        "metadata": {"intent": intent} if intent else {},
    }


def generate_conversation_history() -> Dict:
    """Generate a multi-turn conversation history."""
    conversation_id = str(uuid.uuid4())
    user_id = f"user_{random.randint(10000, 99999)}"
    
    # Determine conversation flow
    flows = [
        # Simple single-turn
        [("return", "I want to return my laptop")],
        # Two-turn billing
        [("billing", "I was charged twice"), ("billing", "Order number is ORD-123456")],
        # Multi-turn technical support
        [("technical", "App keeps crashing"), ("technical", "I tried reinstalling"), 
         ("technical", "Still not working")],
        # Return with escalation
        [("return", "Need to return item"), ("return", "It's been 35 days"), 
         ("escalation", "This is unfair, I want to speak to manager")],
        # General inquiry to support
        [("general_inquiry", "Do you ship to Canada?"), ("general_inquiry", "How long does it take?"),
         ("support", "What about customs fees?")],
    ]
    
    flow = random.choice(flows)
    history = []
    
    for intent, message in flow:
        # User message
        history.append(generate_conversation_turn("user", message, intent))
        # Assistant response
        responses = {
            "return": "I can help you with your return request.",
            "billing": "Let me look into this billing issue for you.",
            "technical": "I'll help troubleshoot this technical problem.",
            "support": "Here's the information about that product.",
            "general_inquiry": "I can help answer that question.",
            "escalation": "I understand your frustration. Let me escalate this.",
            "grievance": "I sincerely apologize for this experience.",
        }
        history.append(generate_conversation_turn("assistant", responses.get(intent, "How can I help?")))
    
    return {
        "conversation_id": conversation_id,
        "user_id": user_id,
        "created_at": (datetime.now() - timedelta(hours=random.randint(1, 48))).isoformat(),
        "updated_at": datetime.now().isoformat(),
        "status": random.choice(["active", "completed", "escalated"]),
        "history": history,
        "primary_intent": flow[0][0],
        "turn_count": len(flow),
    }


def generate_conversations(num_conversations: int = 50) -> List[Dict]:
    """Generate conversation histories."""
    print(f"Generating {num_conversations} conversation histories...")
    
    conversations = []
    for _ in range(num_conversations):
        conversations.append(generate_conversation_history())
    
    return conversations


def generate_feedback_csv(num_records: int = 100) -> List[Dict]:
    """Generate raw feedback data for analytics testing."""
    print(f"Generating {num_records} feedback records...")
    
    positive_comments = [
        "Great service, very helpful!",
        "Quick resolution to my problem",
        "Friendly and knowledgeable agent",
        "Solved my issue on first contact",
        "Easy to use chat interface",
        "Fast response times",
        "Accurate information provided",
        "Exceeded my expectations",
        "Will definitely recommend",
        "Professional and courteous",
        "Problem fixed immediately",
        "Clear instructions provided",
        "Very satisfied with outcome",
        "Saved me a lot of time",
        "Best customer service experience",
    ]
    
    negative_comments = [
        "Not helpful at all",
        "Waste of my time",
        "Didn't understand my problem",
        "Gave wrong information",
        "Took too long to respond",
        "Had to repeat myself multiple times",
        "Couldn't solve my issue",
        "Very frustrating experience",
        "Agent seemed uninformed",
        "Disconnected unexpectedly",
        "No follow up as promised",
        "Billing issue not resolved",
        "Return process unclear",
        "Technical solution didn't work",
        "Rude customer service",
    ]
    
    neutral_comments = [
        "It was okay I guess",
        "Average service",
        "Met basic expectations",
        "Nothing special",
        "Got the job done",
        "Standard support experience",
        "Took a while but resolved",
        "Could be better",
        "Acceptable service",
        "Room for improvement",
    ]
    
    feedback_records = []
    
    for i in range(num_records):
        sentiment = random.choices(
            ["positive", "negative", "neutral"],
            weights=[0.5, 0.35, 0.15]
        )[0]
        
        if sentiment == "positive":
            comment = random.choice(positive_comments)
            rating = random.choice([4, 5])
        elif sentiment == "negative":
            comment = random.choice(negative_comments)
            rating = random.choice([1, 2])
        else:
            comment = random.choice(neutral_comments)
            rating = 3
        
        # Add some typos to feedback
        if random.random() < 0.1:
            comment = apply_typos(comment, typo_prob=0.1)
        
        # Occasionally add intent context
        intent_context = ""
        if random.random() < 0.3:
            intent = random.choice(INTENTS)
            intent_context = f"[{intent}] "
        
        record = {
            "feedback_id": f"FB{random.randint(100000, 999999)}",
            "conversation_id": str(uuid.uuid4()),
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "rating": rating,
            "sentiment": sentiment,
            "comment": intent_context + comment,
            "category": random.choice(["return", "billing", "technical", "general", "other"]),
            "user_id": f"user_{random.randint(10000, 99999)}",
        }
        feedback_records.append(record)
    
    return feedback_records


def save_datasets(
    intents: List[Dict],
    conversations: List[Dict],
    feedback: List[Dict],
    output_dir: Path
):
    """Save all datasets to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save intents as JSON
    intents_file = output_dir / "test_intents.json"
    with open(intents_file, "w") as f:
        json.dump({
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_samples": len(intents),
                "categories": list(set(s["intent"] for s in intents)),
            },
            "samples": intents
        }, f, indent=2)
    print(f"  ✓ Saved {len(intents)} intent samples to {intents_file}")
    
    # Save conversations as JSONL
    conversations_file = output_dir / "test_conversations.jsonl"
    with open(conversations_file, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    print(f"  ✓ Saved {len(conversations)} conversations to {conversations_file}")
    
    # Save feedback as CSV
    feedback_file = output_dir / "test_feedback_raw.csv"
    if feedback:
        with open(feedback_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=feedback[0].keys())
            writer.writeheader()
            writer.writerows(feedback)
    print(f"  ✓ Saved {len(feedback)} feedback records to {feedback_file}")
    
    # Print statistics
    print("\nIntent Distribution:")
    intent_counts = {}
    for s in intents:
        intent_counts[s["intent"]] = intent_counts.get(s["intent"], 0) + 1
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count} ({count/len(intents)*100:.1f}%)")
    
    print("\nConversation Statistics:")
    total_turns = sum(c["turn_count"] for c in conversations)
    print(f"  Total conversations: {len(conversations)}")
    print(f"  Total turns: {total_turns}")
    print(f"  Average turns per conversation: {total_turns/len(conversations):.1f}")
    
    print("\nFeedback Statistics:")
    sentiment_counts = {}
    for f in feedback:
        sentiment_counts[f["sentiment"]] = sentiment_counts.get(f["sentiment"], 0) + 1
    for sentiment, count in sorted(sentiment_counts.items()):
        print(f"  {sentiment}: {count} ({count/len(feedback)*100:.1f}%)")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Synthetic Test Data Generator")
    print("=" * 60)
    print()
    
    # Determine output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir  # Save to same directory as script
    
    # Generate datasets
    intents = generate_intent_dataset(200)
    conversations = generate_conversations(50)
    feedback = generate_feedback_csv(100)
    
    print()
    
    # Save datasets
    save_datasets(intents, conversations, feedback, output_dir)
    
    print()
    print("=" * 60)
    print("Data generation complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
