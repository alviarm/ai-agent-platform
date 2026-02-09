#!/usr/bin/env python3
"""
Seed ChromaDB with synthetic FAQ documents for local testing.
Creates 50+ documents covering returns, billing, tech support, and more.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Synthetic FAQ documents - 50+ documents across categories
SYNTHETIC_FAQS = [
    # === RETURNS (15 documents) ===
    {
        "question": "What is your return policy?",
        "answer": "We accept returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days after we receive the returned item.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "How do I initiate a return?",
        "answer": "To initiate a return, log into your account, go to 'Order History', select the order you want to return, and click 'Return Item'. You'll receive a prepaid return label via email.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "I received a damaged item, what should I do?",
        "answer": "We apologize for the inconvenience. Please contact us immediately with photos of the damaged item and packaging. We'll send a replacement at no extra charge and provide a prepaid return label for the damaged item.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Can I return an item after 30 days?",
        "answer": "Our standard return window is 30 days. However, for defective items or special circumstances, please contact customer service and we'll review your case individually.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Do I have to pay for return shipping?",
        "answer": "For standard returns due to change of mind, return shipping is the customer's responsibility. For defective, damaged, or incorrect items, we provide free prepaid return labels.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "How long does a refund take to process?",
        "answer": "Once we receive your returned item, refunds are typically processed within 5-7 business days. It may take an additional 2-5 business days for the funds to appear in your account depending on your bank.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Can I exchange an item instead of returning it?",
        "answer": "Yes, exchanges are available for different sizes or colors of the same item. During the return process, select 'Exchange' instead of 'Refund' and choose your preferred replacement.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "I lost my receipt, can I still return an item?",
        "answer": "Yes, if you have an account with us, we can look up your order history. For guest checkouts, please provide the email address used for purchase and approximate order date.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Are gift returns accepted?",
        "answer": "Yes, gift recipients can return items for store credit. The gift giver will not be notified of the return. You'll need the order number or gift receipt.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "What items cannot be returned?",
        "answer": "Items that cannot be returned include: personal care products (for hygiene reasons), customized or personalized items, perishable goods, and digital downloads. All sales are final for these categories.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "My return package was lost in transit",
        "answer": "If you used our prepaid return label, file a claim with our support team. We'll investigate with the carrier and process your refund once confirmed. Keep your tracking number for reference.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Can I return items bought with a gift card?",
        "answer": "Yes, returns for gift card purchases are refunded to a new gift card. The new card will be emailed to you within 24 hours of return processing.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "I received the wrong item, how do I return it?",
        "answer": "We sincerely apologize for the error. Please contact us immediately with your order number and a photo of the item received. We'll send the correct item with expedited shipping and provide a prepaid label for the return.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "Can I return a product without the original box?",
        "answer": "While original packaging is preferred, we accept returns without it as long as the item is in resellable condition. A small restocking fee may apply for items without original packaging.",
        "category": "returns",
        "intent": "return"
    },
    {
        "question": "How do I track my return status?",
        "answer": "You can track your return status by logging into your account and visiting the 'Returns' section. You'll see updates from 'In Transit' to 'Received' to 'Refunded'.",
        "category": "returns",
        "intent": "return"
    },
    
    # === BILLING (12 documents) ===
    {
        "question": "When will my refund appear?",
        "answer": "Refunds typically take 5-7 business days to appear in your account after we receive and process your return. The exact timing depends on your bank or credit card company.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "I was charged twice for my order",
        "answer": "We sincerely apologize for this billing error. Please provide us with the transaction IDs or last 4 digits of your card, and we'll investigate immediately. If confirmed, we'll process a refund for the duplicate charge within 24 hours.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "How do I update my payment method?",
        "answer": "You can update your payment method by logging into your account, going to 'Payment Methods' in your profile settings, and adding or editing your payment information.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Why was my credit card declined?",
        "answer": "Credit cards may be declined due to: insufficient funds, incorrect billing address, expired card, or bank security holds. Please verify your information or contact your bank for more details.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Do you accept PayPal or Apple Pay?",
        "answer": "Yes, we accept PayPal, Apple Pay, Google Pay, and all major credit cards (Visa, MasterCard, American Express, Discover). We also accept Shop Pay and Afterpay for installment payments.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "I see an unauthorized charge on my account",
        "answer": "We take unauthorized charges seriously. Please contact us immediately with the transaction details. We'll investigate and if confirmed as fraud, work with your bank to reverse the charge immediately.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Can I get a price adjustment if the item goes on sale?",
        "answer": "We offer price adjustments within 14 days of purchase if the item goes on sale. Contact customer service with your order number and we'll refund the difference to your original payment method.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "How do I get a copy of my invoice?",
        "answer": "Invoices can be downloaded from your account under 'Order History'. Click on the order number and select 'Download Invoice'. They're also emailed to you at the time of purchase.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Why is there a pending charge on my account?",
        "answer": "Pending charges are authorization holds placed by your bank when you place an order. They typically drop off within 3-5 business days if the order is canceled, or convert to actual charges when shipped.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Can I split my payment between two cards?",
        "answer": "Currently, we only accept single payment methods per order. For purchases exceeding your card limit, consider using PayPal which may allow combining payment sources.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Do you offer payment plans?",
        "answer": "Yes, we partner with Afterpay and Klarna to offer interest-free installment payments. Simply select these options at checkout. Your first payment is due at purchase, with remaining payments every two weeks.",
        "category": "billing",
        "intent": "billing"
    },
    {
        "question": "Why was I charged sales tax?",
        "answer": "Sales tax is collected based on your shipping address and local tax laws. The tax rate is calculated automatically at checkout. Tax-exempt organizations can apply for exemption by submitting your certificate.",
        "category": "billing",
        "intent": "billing"
    },
    
    # === TECHNICAL SUPPORT (13 documents) ===
    {
        "question": "The app keeps crashing when I try to login",
        "answer": "We're sorry you're experiencing this issue. Please try these steps: 1) Force close the app and reopen it, 2) Clear the app cache in your device settings, 3) Update to the latest app version, 4) If problems persist, try uninstalling and reinstalling the app.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "I forgot my password, how do I reset it?",
        "answer": "Click 'Forgot Password' on the login screen and enter your email address. We'll send you a password reset link that expires in 24 hours. Check your spam folder if you don't see the email within a few minutes.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "Two-factor authentication is not working",
        "answer": "If you're not receiving 2FA codes, please check: 1) Your phone has signal/service, 2) The phone number in your account is correct, 3) You're not blocking messages from our short code. If issues persist, contact support to verify your account identity.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "I can't log into my account",
        "answer": "Login issues can be caused by: incorrect password (try resetting), account lockout after failed attempts (wait 30 minutes), or browser cache issues (clear cookies). If none work, contact support for account assistance.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "The website is not loading properly",
        "answer": "Try these troubleshooting steps: 1) Clear your browser cache and cookies, 2) Disable browser extensions temporarily, 3) Try an incognito/private window, 4) Check your internet connection, 5) Try a different browser.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "My account was hacked, what do I do?",
        "answer": "Immediately change your password and enable two-factor authentication if not already active. Review recent orders for unauthorized purchases and contact us immediately so we can secure your account and investigate.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "How do I unsubscribe from marketing emails?",
        "answer": "Click the 'Unsubscribe' link at the bottom of any marketing email. You can also manage email preferences in your account under 'Communication Preferences'. Note: Order-related emails cannot be unsubscribed.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "The product manual download link is broken",
        "answer": "We apologize for the technical issue. Please try refreshing the page or using a different browser. If the problem persists, contact support with the product model number and we'll email you the manual directly.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "My saved payment methods disappeared",
        "answer": "For security, we periodically require re-verification of saved payment methods. Please re-add your payment method in account settings. This is a normal security measure to protect your financial information.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "The mobile app won't install on my phone",
        "answer": "Ensure your device meets the minimum requirements: iOS 13+ or Android 8+. Check available storage space (app requires 100MB). If issues persist, try downloading from the official app store rather than third-party sources.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "I'm getting error code 500 when checking out",
        "answer": "Error 500 indicates a server issue. Please wait a few minutes and try again. If the error persists, clear your cart and re-add items. For urgent orders, try checking out as a guest or contact support for assistance.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "How do I update my email address?",
        "answer": "Go to Account Settings > Profile and click 'Edit' next to your email. You'll need to verify the new email address via a confirmation link we'll send. For security, you'll also need to enter your password.",
        "category": "technical",
        "intent": "technical"
    },
    {
        "question": "Push notifications are not working",
        "answer": "Check notification permissions in your device settings for our app. Ensure 'Do Not Disturb' is off. In the app, go to Settings > Notifications and verify they're enabled. Try logging out and back in.",
        "category": "technical",
        "intent": "technical"
    },
    
    # === GENERAL SUPPORT (10 documents) ===
    {
        "question": "How do I set up my new device?",
        "answer": "Your device comes with a quick start guide in the box. You can also download the full user manual from our website. For video tutorials, visit our YouTube channel or the 'Support' section of our website.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "What are the product specifications?",
        "answer": "Detailed product specifications are available on each product page under the 'Specifications' tab. If you need additional technical details, please contact our product support team.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Is this product compatible with my device?",
        "answer": "Compatibility information is listed on each product page. You can also use our compatibility checker tool on the product page by entering your device model. If you're unsure, contact support with your device details.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Do you have a warranty on products?",
        "answer": "All products come with a standard 1-year manufacturer warranty covering defects. Extended warranties are available for purchase on select items. Warranty claims can be initiated through your account or by contacting support.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "How do I contact customer service by phone?",
        "answer": "You can reach our customer service team at 1-800-SUPPORT (1-800-787-7688). Hours are Monday-Friday 9am-8pm EST, Saturday 10am-6pm EST. We also offer 24/7 support via chat and email.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Where can I find product reviews?",
        "answer": "Customer reviews are available on each product page below the product description. You can filter reviews by rating, date, and verified purchase status. We encourage all customers to leave honest feedback.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Do you offer bulk or wholesale pricing?",
        "answer": "Yes, we offer bulk discounts for orders of 10+ units of the same item. For wholesale inquiries (50+ units), please contact our B2B team at wholesale@example.com for custom pricing.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Can I get a product demo before buying?",
        "answer": "We offer virtual product demos for select high-value items. Schedule a demo through the product page or contact sales. Some items also have 30-day trial periods with full refund if not satisfied.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "How do I redeem a promo code?",
        "answer": "Enter your promo code in the 'Promo Code' box at checkout and click 'Apply'. The discount will be reflected in your order total. Promo codes cannot be combined unless specifically stated.",
        "category": "support",
        "intent": "support"
    },
    {
        "question": "Do you price match competitors?",
        "answer": "Yes, we price match identical items from authorized retailers. Provide proof of the lower price (link or screenshot) and we'll match it plus give you an additional 5% discount. Some exclusions apply.",
        "category": "support",
        "intent": "support"
    },
    
    # === GENERAL INQUIRY (5 documents) ===
    {
        "question": "What are your store hours?",
        "answer": "Our online store is available 24/7. Physical store hours vary by location. You can find specific hours using our store locator on the website.",
        "category": "general",
        "intent": "general_inquiry"
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Yes, we ship to over 100 countries worldwide. International shipping costs and delivery times vary by destination. You can see available shipping options and costs at checkout.",
        "category": "general",
        "intent": "general_inquiry"
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 3-5 business days. Express shipping (1-2 business days) is available for most locations. International shipping typically takes 7-14 business days depending on the destination.",
        "category": "general",
        "intent": "general_inquiry"
    },
    {
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and visiting the 'Order History' page. Click on the order number to see tracking information. We also send tracking updates via email when your order ships.",
        "category": "general",
        "intent": "general_inquiry"
    },
    {
        "question": "Where is my order confirmation email?",
        "answer": "Order confirmations are sent immediately after purchase. If you don't see it within 15 minutes, check your spam/junk folder. If still missing, verify your email address in account settings or contact support.",
        "category": "general",
        "intent": "general_inquiry"
    },
    
    # === ESCALATION/GRIEVANCE (5 documents) ===
    {
        "question": "I want to file a complaint",
        "answer": "We take your concerns seriously. Please describe your issue in detail, including any relevant order numbers or dates. A customer service manager will review your complaint and respond within 24-48 hours.",
        "category": "escalation",
        "intent": "grievance"
    },
    {
        "question": "This is the third time I've contacted you",
        "answer": "I sincerely apologize that your issue hasn't been resolved. I'm escalating this to our resolution team lead who will personally handle your case. You should receive a response within 4 hours.",
        "category": "escalation",
        "intent": "escalation"
    },
    {
        "question": "I need to speak to a supervisor immediately",
        "answer": "I understand you need to speak with a supervisor. I'm connecting you now. If a supervisor is not immediately available, I can schedule a callback within the next 30 minutes. Would that work for you?",
        "category": "escalation",
        "intent": "escalation"
    },
    {
        "question": "I'm extremely dissatisfied with your service",
        "answer": "I sincerely apologize for falling short of your expectations. I'd like to make this right. Please share the details of your experience, and I'll ensure this is escalated to management for immediate resolution.",
        "category": "escalation",
        "intent": "grievance"
    },
    {
        "question": "I want to cancel my membership immediately",
        "answer": "I can help you cancel your membership. Before we proceed, may I ask what led to this decision? Your feedback helps us improve. If you proceed, cancellation takes effect at the end of your current billing cycle.",
        "category": "escalation",
        "intent": "grievance"
    },
]


def get_chroma_client():
    """Initialize ChromaDB client."""
    chroma_host = os.environ.get("CHROMADB_HOST", "localhost")
    chroma_port = os.environ.get("CHROMADB_PORT", "8000")
    
    # Try HTTP client first (for Docker), fallback to persistent client
    try:
        client = chromadb.HttpClient(
            host=chroma_host,
            port=int(chroma_port),
        )
        # Test connection
        client.heartbeat()
        print(f"Connected to ChromaDB at {chroma_host}:{chroma_port}")
        return client
    except Exception as e:
        print(f"HTTP client failed ({e}), using persistent client...")
        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=persist_dir)


def seed_database():
    """Seed ChromaDB with FAQ documents."""
    print("=" * 60)
    print("Seeding ChromaDB with FAQ Documents")
    print("=" * 60)
    
    # Get client
    client = get_chroma_client()
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name="faq_documents",
        metadata={"hnsw:space": "cosine"},
    )
    
    # Clear existing data if any
    existing_count = collection.count()
    if existing_count > 0:
        print(f"Collection has {existing_count} documents, skipping seed (already populated)")
        return
    
    # Load embedding model
    print("Loading embedding model...")
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(model_name)
    
    # Prepare documents
    documents = []
    metadatas = []
    ids = []
    
    for i, faq in enumerate(SYNTHETIC_FAQS):
        # Combine question and answer for embedding
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        documents.append(content)
        metadatas.append({
            "question": faq["question"],
            "answer": faq["answer"],
            "category": faq["category"],
            "intent": faq["intent"],
            "doc_id": f"faq_{i:03d}",
        })
        ids.append(f"faq_{i:03d}")
    
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = embedding_model.encode(documents).tolist()
    
    # Add to collection in batches
    batch_size = 50
    for i in range(0, len(documents), batch_size):
        batch_end = min(i + batch_size, len(documents))
        collection.add(
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end],
        )
        print(f"  Added batch {i//batch_size + 1}: documents {i+1}-{batch_end}")
    
    print(f"\n✓ Successfully seeded {collection.count()} documents")
    
    # Print statistics by category
    print("\nDocument Statistics:")
    all_docs = collection.get()
    categories = {}
    for meta in all_docs["metadatas"]:
        cat = meta.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")
    
    # Test query
    print("\n" + "=" * 60)
    print("Running test query...")
    test_query = "How do I return a damaged item?"
    query_embedding = embedding_model.encode([test_query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    
    print(f"Query: '{test_query}'")
    print("Top 3 results:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        score = 1 - dist
        print(f"  {i+1}. [{meta['category']}] Score: {score:.3f}")
        print(f"     Q: {meta['question']}")
    
    print("\n✓ ChromaDB seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    seed_database()
