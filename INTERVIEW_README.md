# Customer Service AI Platform - Interview Demo Guide

> **Repository:** https://github.com/alviarm/ai-agent-platform  
> **Status:** ✅ MVP Ready for Interview Demonstration

---

## 🎯 Quick Demo Commands

```bash
# Clone and setup
git clone https://github.com/alviarm/ai-agent-platform.git
cd ai-agent-platform
pip install -r requirements.txt

# Quick component test
python3 -c "from src.intent_classifier import IntentClassifier; \
    clf = IntentClassifier(); \
    print(clf.predict('I want to return my laptop'))"

# Start API server
python -m src.api.main

# Run tests
make test
```

---

## ✅ MVP Verification Status

### Core Components

| Component | Status | Demo Capability |
|-----------|--------|-----------------|
| **Intent Classifier** | ✅ Working | Classifies 7 intent categories in real-time |
| **Vector Store** | ✅ Working | ChromaDB with 50+ FAQ documents |
| **RAG Engine** | ✅ Working | Retrieval + Generation pipeline |
| **Feedback Analyzer** | ✅ Working | Sentiment analysis + Topic modeling |
| **FastAPI Server** | ✅ Working | REST API with /chat, /classify, /feedback endpoints |

### Testing Infrastructure

| Test Suite | Status | Coverage |
|------------|--------|----------|
| **Unit Tests** | ✅ Ready | Intent, RAG, Analytics components |
| **Integration Tests** | ✅ Ready | E2E customer journey tests |
| **Local Infrastructure** | ✅ Ready | Docker Compose with LocalStack, ChromaDB, Redis |
| **Synthetic Data** | ✅ Generated | 200+ intent samples, 50 conversations, 100 feedback records |

---

## 🚀 Demonstration Scenarios

### Scenario 1: Intent Classification
```python
from src.intent_classifier import IntentClassifier

clf = IntentClassifier()
queries = [
    "I want to return my laptop",
    "My bill is wrong, I was charged twice",
    "The app keeps crashing when I login",
    "I need to speak to a supervisor immediately",
]

for query in queries:
    result = clf.predict(query)[0]
    print(f"{query[:40]}... -> {result['intent']} ({result['confidence']:.1%})")
```

### Scenario 2: RAG Response Generation
```python
from src.response_generator import RAGEngine, VectorStore, ConversationManager

# Initialize components
vs = VectorStore(persist_dir='./data/chroma')
cm = ConversationManager(use_dynamodb=False)
rag = RAGEngine(vector_store=vs, conversation_manager=cm)

# Generate response
result = rag.generate_response(
    query="How do I return a damaged item?",
    intent="return"
)

print(f"Response: {result['response']}")
print(f"Retrieved {len(result['retrieved_context'])} context chunks")
```

### Scenario 3: Feedback Analysis
```python
from src.feedback_pipeline import FeedbackAnalyzer

analyzer = FeedbackAnalyzer()

# Analyze sentiment
result = analyzer.analyze_sentiment("Great service, very helpful!")
print(f"Sentiment: {result['label']} ({result['confidence']:.1%})")

# Extract keywords
keywords = analyzer.extract_keywords_rake(
    "The customer service was excellent and very helpful", 
    num_keywords=5
)
print(f"Keywords: {[k[1] for k in keywords]}")
```

---

## 🐳 Docker Demo (Recommended for Interviews)

```bash
# Start full local infrastructure
docker-compose -f docker-compose.test.yml up -d localstack chromadb redis

# Seed with test data
python tests/scripts/seed_chroma.py

# Start API server
docker-compose -f docker-compose.test.yml --profile api up api-server

# API is now available at http://localhost:8002
```

### API Endpoints for Demo

| Endpoint | Method | Demo Usage |
|----------|--------|------------|
| `/health` | GET | Check all services are up |
| `/classify` | POST | Classify intent of customer message |
| `/chat` | POST | Full chat with RAG response |
| `/feedback` | POST | Submit feedback for analysis |
| `/analytics` | GET | View analytics summary |

**Example API Call:**
```bash
curl -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to return my laptop"}'
```

---

## 📊 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   /classify  │  │    /chat     │  │    /feedback     │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼───────────────────┼────────────┘
          │                 │                   │
    ┌─────▼─────┐    ┌──────▼──────┐   ┌───────▼────────┐
    │   BERT    │    │    RAG      │   │  NLP Pipeline  │
    │ Classifier│    │   Engine    │   │   (VADER/TF)   │
    └─────┬─────┘    └──────┬──────┘   └───────┬────────┘
          │                 │                   │
    ┌─────▼─────┐    ┌──────▼──────┐   ┌───────▼────────┐
    │  PyTorch  │    │  ChromaDB   │   │  Topic Model   │
    │  Models   │    │ Vector Store│   │    (LDA)       │
    └───────────┘    └─────────────┘   └────────────────┘
```

---

## 🧪 Testing for Interviews

```bash
# Run all tests
make test

# Run specific test suites
make test-unit          # Component tests
make test-integration   # E2E tests
make test-cov          # With coverage report

# Quick smoke test
python -m pytest tests/unit/test_classifier.py::TestIntentClassifier::test_predict_single_text -v
```

---

## 📁 Repository Structure

```
ai-agent-platform/
├── src/                          # Core source code
│   ├── intent_classifier/        # BERT-based classification
│   ├── response_generator/       # RAG with ChromaDB
│   ├── feedback_pipeline/        # NLP analytics
│   ├── api/main.py              # FastAPI server
│   └── infrastructure/           # AWS CDK
├── tests/                        # Complete test suite
│   ├── unit/                     # Component tests
│   ├── integration/              # E2E tests
│   ├── data/                     # Synthetic test data
│   └── scripts/                  # Test helpers
├── lambda_functions/             # AWS Lambda containers
├── docker-compose.test.yml       # Local infrastructure
├── Makefile                      # Dev commands
└── template.yaml                 # AWS SAM
```

---

## 🔑 Key Features to Highlight

1. **Production-Ready Architecture**
   - Modular design with separation of concerns
   - Configurable for local dev or AWS deployment
   - Comprehensive error handling and logging

2. **AI/ML Capabilities**
   - State-of-the-art BERT for intent classification
   - RAG (Retrieval-Augmented Generation) for contextual responses
   - NLP pipeline with sentiment analysis and topic modeling

3. **Testing Infrastructure**
   - 100% offline testing with LocalStack
   - Synthetic data generation for robust testing
   - E2E tests simulating real customer journeys

4. **DevOps Ready**
   - Docker containerization
   - AWS SAM for serverless deployment
   - CDK for infrastructure as code

---

## ⚠️ Known Limitations (Interview Context)

| Limitation | Explanation | Workaround |
|------------|-------------|------------|
| Model not trained | DistilBERT used in default mode | Works with pre-trained weights; training script included |
| LLM requires API key | Uses LiteLLM for generation | Falls back to template responses without API key |
| ChromaDB version | Updated for v0.4+ compatibility | Backwards compatible fallback included |

---

## 💻 System Requirements

- Python 3.9+
- 8GB RAM (for local model inference)
- Docker (for full local infrastructure)
- Git

---

## 📞 Interview Demo Checklist

- [ ] Clone repository
- [ ] Run `pip install -r requirements.txt`
- [ ] Run `python -c "from src.intent_classifier import IntentClassifier; print('✅ Working')"`
- [ ] Run `make test` (or `pytest tests/unit/ -v`)
- [ ] Start API: `python -m src.api.main`
- [ ] Test API: `curl http://localhost:8000/health`
- [ ] Show Docker Compose: `docker-compose -f docker-compose.test.yml up -d`

---

**Last Updated:** 2026-02-09  
**Version:** 0.1.0 MVP  
**Status:** ✅ Ready for Interview Demonstration
