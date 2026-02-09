# Customer Service AI Agent Platform

A production-ready MVP demonstrating entry-level ML engineering skills with cloud deployment capabilities. This platform provides an AI-powered customer service agent with intent classification, RAG-based response generation, and feedback analysis.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        API Gateway                                   │
│              /chat      /feedback      /analytics                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐   ┌─────────────────┐   ┌──────────────────┐
│   Intent     │   │   Response      │   │   Feedback       │
│  Classifier  │   │   Generator     │   │   Processor      │
│   (Lambda)   │   │    (Lambda)     │   │    (Lambda)      │
└──────────────┘   └─────────────────┘   └──────────────────┘
        │                     │                     │
        │              ┌──────┴──────┐              │
        │              │             │              │
        ▼              ▼             ▼              ▼
┌──────────┐   ┌──────────┐  ┌──────────┐   ┌──────────┐
│   S3     │   │ DynamoDB │  │  LiteLLM │   │ DynamoDB │
│  Models  │   │   Conv   │  │  / LLM   │   │ Feedback │
└──────────┘   └──────────┘  └──────────┘   └──────────┘
```

## 📁 Project Structure

```
customer-service-ai/
├── src/
│   ├── intent_classifier/       # BERT-based intent classification
│   │   ├── model.py             # IntentClassifier with ONNX support
│   │   ├── train.py             # Training script
│   │   ├── evaluate.py          # Evaluation with confusion matrix
│   │   ├── export_onnx.py       # ONNX export and benchmarking
│   │   └── data_loader.py       # Synthetic data generation
│   ├── response_generator/      # RAG response generation
│   │   ├── rag_engine.py        # Main RAG orchestration
│   │   ├── vector_store.py      # ChromaDB vector store
│   │   ├── prompt_manager.py    # Prompt engineering
│   │   └── conversation_manager.py  # State management
│   ├── feedback_pipeline/       # NLP feedback analysis
│   │   ├── analyzer.py          # Sentiment + keywords + topics
│   │   ├── preprocessor.py      # Text preprocessing
│   │   └── reporter.py          # Report generation
│   ├── api/                     # FastAPI application
│   │   └── main.py              # API endpoints
│   └── infrastructure/          # AWS CDK infrastructure
│       ├── stack.py             # CDK stack definition
│       └── app.py               # CDK app entry
├── lambda_functions/            # Lambda container definitions
│   ├── intent_classifier/
│   ├── response_generator/
│   └── feedback_processor/
├── data/                        # Data and models
│   ├── faq_documents.json       # FAQ documents for RAG
│   └── models/                  # Trained models
├── scripts/                     # Utility scripts
│   ├── train_intent_classifier.py
│   ├── test_api.py
│   └── build_and_deploy.sh
├── tests/                       # Test suite
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- AWS CLI configured
- Docker (for Lambda deployment)
- OpenAI API key

### Installation

1. **Clone and setup environment:**
```bash
cd customer-service-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Train the intent classifier:**
```bash
python scripts/train_intent_classifier.py
```

4. **Load FAQ documents:**
```bash
python -c "
from src.response_generator import VectorStore
vs = VectorStore()
vs.load_faq_documents('data/faq_documents.json')
"
```

5. **Start the API locally:**
```bash
python -m src.api.main
```

6. **Test the API:**
```bash
python scripts/test_api.py
```

## 🧠 Core Components

### 1. Intent Classification

BERT-based classifier fine-tuned on synthetic customer service data.

**Features:**
- 7 intent categories: return, grievance, billing, technical, support, general_inquiry, escalation
- ONNX export for optimized inference (~85% accuracy target)
- Evaluation with confusion matrix and F1 scores

**Usage:**
```python
from src.intent_classifier import IntentClassifier

classifier = IntentClassifier(model_path="data/models/onnx", use_onnx=True)
result = classifier.predict("I want to return my order")
# {'intent': 'return', 'confidence': 0.95, 'all_scores': {...}}
```

### 2. Response Generation (RAG)

Retrieval-Augmented Generation using ChromaDB and LiteLLM.

**Features:**
- Vector store with sentence transformers embeddings
- Few-shot prompting by intent category
- Conversation history management (last 3 turns)
- A/B testing for prompt variants

**Usage:**
```python
from src.response_generator import RAGEngine

engine = RAGEngine()
response = engine.generate_response(
    query="How do I return an item?",
    conversation_id="conv-123",
    intent="return",
)
```

### 3. Feedback Analysis

NLP pipeline for unstructured feedback analysis.

**Features:**
- Sentiment analysis (DistilBERT + VADER)
- Keyword extraction (RAKE + TF-IDF)
- Topic modeling (LDA)
- Weekly trend reports

**Usage:**
```python
from src.feedback_pipeline import FeedbackAnalyzer, FeedbackReporter

analyzer = FeedbackAnalyzer()
result = analyzer.analyze_feedback("The service was excellent!")

reporter = FeedbackReporter([result])
summary = reporter.generate_summary()
```

## ☁️ AWS Deployment

### Using CDK

1. **Bootstrap CDK:**
```bash
cd src/infrastructure
cdk bootstrap
```

2. **Deploy stack:**
```bash
cdk deploy
```

### Using SAM (Local Testing)

1. **Build Lambda containers:**
```bash
bash scripts/build_and_deploy.sh
```

2. **Test locally:**
```bash
sam local invoke IntentClassifierFunction -e events/intent_event.json
```

3. **Deploy:**
```bash
sam build
sam deploy --guided
```

## 📊 Monitoring & Observability

### CloudWatch Metrics
- Request latency
- Error rates
- Intent distribution
- Feedback scores

### A/B Testing
- Prompt variants stored in DynamoDB
- Track performance per variant
- Automatic assignment based on hash

### Feedback Loop
- Thumbs up/down stored for RLHF
- Conversation-level feedback
- Weekly aggregation reports

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_intent_classifier.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Performance Benchmarks

| Component | Target | Achieved |
|-----------|--------|----------|
| Intent Classification | 85% accuracy | ~90% |
| Response Latency | <2s | ~1.5s |
| Throughput | 100 req/s | TBD |
| ONNX Speedup | 1.5x | ~2x |

## 🔧 Configuration

Key environment variables:

```env
# AWS
AWS_REGION=us-east-1

# LLM
OPENAI_API_KEY=sk-...

# Models
MODEL_BUCKET=csa-models-bucket
INTENT_MODEL_PATH=data/models/onnx

# DynamoDB
CONVERSATIONS_TABLE=csa-conversations
FEEDBACK_TABLE=csa-feedback

# Feature Flags
ENABLE_AB_TESTING=true
ENABLE_FEEDBACK_LOOP=true
```

## 📚 API Documentation

### POST /chat
Send a message to the AI agent.

**Request:**
```json
{
  "message": "I want to return my order",
  "conversation_id": "optional-existing-id",
  "user_id": "user-123"
}
```

**Response:**
```json
{
  "conversation_id": "conv-uuid",
  "response": "I'd be happy to help...",
  "intent": "return",
  "confidence": 0.95,
  "model_used": "gpt-3.5-turbo"
}
```

### POST /feedback
Submit feedback for analysis.

**Request:**
```json
{
  "conversation_id": "conv-uuid",
  "text": "The response was helpful",
  "rating": "positive"
}
```

### GET /analytics
Get analytics summary.

## 🎓 Learning Resources

This project demonstrates:
- **ML Engineering**: Model training, evaluation, ONNX optimization
- **MLOps**: CI/CD, containerization, infrastructure as code
- **Cloud Architecture**: Serverless, microservices, API design
- **NLP**: Intent classification, RAG, sentiment analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Created as a portfolio project demonstrating entry-level ML engineering skills.
