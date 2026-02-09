"""Configuration for Response Generation."""

from typing import Dict, List

# Vector Store Configuration
VECTOR_STORE_PATH = "data/vector_store"
CHROMA_PERSIST_DIR = "data/chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# LLM Configuration
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
FALLBACK_MODEL = "claude-instant-1"
MAX_TOKENS = 500
TEMPERATURE = 0.7
TOP_P = 0.9

# Conversation Management
MAX_HISTORY_TURNS = 3
CONVERSATION_TIMEOUT_MINUTES = 30

# Few-shot examples by intent
FEW_SHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "return": [
        {
            "user": "I want to return my order",
            "assistant": "I'd be happy to help you with your return. Could you please provide your order number? You can find it in your confirmation email.",
        },
        {
            "user": "The product arrived damaged",
            "assistant": "I'm sorry to hear your item arrived damaged. We can definitely process a return or replacement for you. Do you have photos of the damage, and what's your order number?",
        },
    ],
    "grievance": [
        {
            "user": "This is unacceptable service!",
            "assistant": "I sincerely apologize for the frustration you've experienced. I understand this situation has been disappointing, and I want to make it right. Let me escalate this to a supervisor who can address your concerns properly.",
        },
        {
            "user": "I've contacted you 5 times with no resolution",
            "assistant": "I'm truly sorry that your issue hasn't been resolved despite multiple attempts. This falls short of our standards. I'm flagging this for immediate escalation to ensure you receive the attention you deserve.",
        },
    ],
    "billing": [
        {
            "user": "I was charged twice",
            "assistant": "I apologize for the duplicate charge. Let me look into this right away. Could you provide the transaction IDs or the last 4 digits of the card used? I'll initiate a refund for the duplicate charge immediately.",
        },
        {
            "user": "When will my refund appear?",
            "assistant": "Refunds typically take 5-7 business days to appear in your account, depending on your bank. If it has been longer than that, I can check the status for you. When was the refund processed?",
        },
    ],
    "technical": [
        {
            "user": "The app keeps crashing",
            "assistant": "I'm sorry you're experiencing crashes. Let's troubleshoot this. First, try clearing the app cache and ensuring you have the latest version installed. What device and OS version are you using?",
        },
        {
            "user": "I can't log in to my account",
            "assistant": "Let's get you back into your account. Have you tried resetting your password? If you're not receiving the reset email, please check your spam folder and ensure you're using the correct email address.",
        },
    ],
    "support": [
        {
            "user": "How do I set up my new device?",
            "assistant": "I'd be happy to help you set up your new device! You can find the complete setup guide in the user manual that came with the product, or I can walk you through the basic steps. Which model do you have?",
        },
        {
            "user": "What are the product specifications?",
            "assistant": "You can find detailed specifications on the product page. Would you like me to highlight any specific specifications you're interested in, such as dimensions, weight, or technical capabilities?",
        },
    ],
    "general_inquiry": [
        {
            "user": "What are your store hours?",
            "assistant": "Our online store is available 24/7! For physical locations, hours vary by store. You can find specific hours using our store locator on the website. Is there a particular location you're asking about?",
        },
        {
            "user": "Do you ship internationally?",
            "assistant": "Yes, we ship to over 100 countries worldwide! Shipping costs and delivery times vary by destination. You can see available shipping options and costs at checkout after entering your address.",
        },
    ],
    "escalation": [
        {
            "user": "This is urgent, I need help now",
            "assistant": "I understand this is urgent and I'm prioritizing your request immediately. I'm connecting you with our priority support team who can provide immediate assistance. Please hold for just a moment.",
        },
        {
            "user": "I'm a premium customer",
            "assistant": "Thank you for being a premium customer. I'm routing you to our premium support specialist who can provide expedited service. Your case has been marked as high priority.",
        },
    ],
}

# System prompts by intent
INTENT_SYSTEM_PROMPTS: Dict[str, str] = {
    "return": """You are a helpful customer service assistant specializing in returns and refunds.
Be empathetic, clear about the process, and helpful in resolving the customer's return request.
Key points to address: return window, condition requirements, refund method, and timeline.""",
    
    "grievance": """You are a senior customer service representative handling escalated complaints.
Show empathy, acknowledge the customer's frustration, apologize sincerely, and offer concrete solutions.
Always offer to escalate to a supervisor if the customer remains unsatisfied.""",
    
    "billing": """You are a billing specialist helping with payment and refund issues.
Be precise with amounts, timelines, and processes. Verify details before making changes.
Provide clear next steps for any billing adjustments.""",
    
    "technical": """You are a technical support specialist helping customers with product issues.
Provide step-by-step troubleshooting guidance. Be patient and clear.
If the issue requires advanced support, offer to create a ticket for the engineering team.""",
    
    "support": """You are a product support specialist providing helpful information.
Be informative, clear, and guide customers to the right resources.
Offer product recommendations when appropriate.""",
    
    "general_inquiry": """You are a friendly customer service representative answering general questions.
Be helpful, concise, and direct customers to appropriate resources when needed.""",
    
    "escalation": """You are handling a priority/escalated case requiring immediate attention.
Acknowledge urgency, take ownership, and provide expedited resolution paths.
Connect to appropriate specialized teams immediately.""",
}

DEFAULT_SYSTEM_PROMPT = """You are a helpful customer service AI assistant. 
Be professional, empathetic, and solution-oriented.
Provide accurate information and clear next steps."""
