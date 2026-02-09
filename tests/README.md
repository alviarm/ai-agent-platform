# Customer Service AI Platform - Local Testing Suite

A complete local testing environment for validating the Customer Service AI Agent Platform MVP in an isolated, offline/demo environment.

## Overview

This testing suite provides:
- **Local Infrastructure Simulation** via Docker Compose (LocalStack, ChromaDB, Redis)
- **Synthetic Test Data** generation (200+ intent samples, 50+ conversations, 100+ feedback records)
- **Component Tests** for Intent Classifier, RAG Engine, and Feedback Analytics
- **Integration & E2E Tests** simulating complete customer journeys
- **Zero AWS Dependencies** - everything runs 100% locally

## Quick Start

### 1. Start Local Infrastructure

```bash
cd customer-service-ai

# Start all services (LocalStack, ChromaDB, Redis)
docker-compose -f docker-compose.test.yml up -d localstack chromadb redis

# Wait for services to be ready (about 30 seconds)
sleep 30

# Verify services are running
docker-compose -f docker-compose.test.yml ps
```

### 2. Seed Test Data

```bash
# Generate synthetic test data
python tests/data/generate_synthetic_data.py

# Seed ChromaDB with FAQ documents
python tests/scripts/seed_chroma.py
```

### 3. Run Tests

```bash
# Run all tests
cd customer-service-ai
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Skip slow tests
python -m pytest tests/ -m "not slow"
```

## Docker Compose Testing (Recommended)

Run everything in isolated containers:

```bash
# Run the full test suite in Docker
docker-compose -f docker-compose.test.yml up test-runner

# Start the API server for manual testing
docker-compose -f docker-compose.test.yml --profile api up api-server

# View logs
docker-compose -f docker-compose.test.yml logs -f test-runner
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                     # Pytest configuration
├── README.md                      # This file
├── data/
│   ├── generate_synthetic_data.py # Synthetic data generator
│   ├── test_intents.json          # Generated intent samples
│   ├── test_conversations.jsonl   # Generated conversations
│   └── test_feedback_raw.csv      # Generated feedback data
├── scripts/
│   ├── init_localstack.sh         # LocalStack initialization
│   ├── seed_chroma.py             # ChromaDB seeding script
│   └── wait-for-services.sh       # Service health check script
├── unit/
│   ├── test_classifier.py         # Intent classifier tests
│   ├── test_rag.py                # RAG engine tests
│   └── test_analytics.py          # Feedback analytics tests
└── integration/
    └── test_e2e.py                # End-to-end integration tests
```

## Component Tests

### Intent Classifier (`test_classifier.py`)

Tests the BERT-based intent classification:

- ✅ Model initialization (PyTorch and ONNX)
- ✅ Single and batch prediction
- ✅ Classification accuracy ≥ 85%
- ✅ Inference latency < 200ms
- ✅ Confidence thresholding for escalation
- ✅ Handling of typos and edge cases
- ✅ ONNX/PyTorch parity

### RAG Engine (`test_rag.py`)

Tests the Retrieval-Augmented Generation pipeline:

- ✅ Document retrieval accuracy ≥ 80%
- ✅ Intent-based filtering
- ✅ Prompt template rendering
- ✅ Conversation history truncation (max 3 turns)
- ✅ Multi-turn conversation persistence
- ✅ Fallback response on LLM failure

### Feedback Analytics (`test_analytics.py`)

Tests the NLP feedback analysis pipeline:

- ✅ Sentiment analysis (VADER/transformer)
- ✅ Sentiment distribution validation
- ✅ RAKE keyword extraction
- ✅ TF-IDF keyword extraction
- ✅ LDA topic modeling (3-5 topics, coherence > 0.4)
- ✅ Weekly trend aggregation
- ✅ Category-based aggregation

## Integration Tests

### End-to-End Customer Journey (`test_e2e.py`)

Simulates complete customer interactions:

1. **Billing Issue Journey**
   - Customer: "My bill is wrong, I was charged twice"
   - System: Classifies as `billing`, provides relevant response
   - Customer: Provides negative feedback
   - System: Feedback appears in analytics

2. **Return Request Journey**
   - Customer: "I received a damaged laptop and need to return it"
   - System: Classifies as `return`, retrieves relevant FAQ
   - Customer: Follow-up questions about shipping
   - Customer: Provides positive feedback

3. **API Endpoint Tests**
   - Health check endpoint
   - Intent classification endpoint
   - Chat endpoint
   - Feedback endpoint
   - Analytics endpoint
   - Conversation history endpoint

### LocalStack Integration

Tests AWS service simulation:

- ✅ DynamoDB tables creation
- ✅ S3 bucket operations
- ✅ CloudWatch log groups
- ✅ Conversation CRUD operations

### ChromaDB Integration

Tests vector store operations:

- ✅ Connection to ChromaDB
- ✅ Document search and retrieval
- ✅ Embedding similarity search

## Synthetic Test Data

### Intent Classification Samples (`test_intents.json`)

200 labeled samples across 7 categories:
- `return` (30 samples)
- `billing` (30 samples)
- `technical` (30 samples)
- `support` (30 samples)
- `general_inquiry` (30 samples)
- `escalation` (25 samples)
- `grievance` (25 samples)

Includes:
- Clean samples (80%)
- Samples with typos (15%)
- Ambiguous queries (5%)

### Conversation Histories (`test_conversations.jsonl`)

50 multi-turn conversations with:
- Single-turn queries
- Multi-turn support interactions
- Escalation scenarios
- Mixed intent flows

### Feedback Records (`test_feedback_raw.csv`)

100 feedback records with:
- Sentiment labels (positive/negative/neutral)
- Ratings (1-5 stars)
- Categories (return/billing/technical/general)
- Timestamps for trend analysis

## Environment Configuration

### Local Development

```bash
# Set environment variables
export TEST_MODE=true
export AWS_ENDPOINT_URL=http://localhost:4566
export CHROMADB_HOST=localhost
export CHROMADB_PORT=8001
export REDIS_HOST=localhost
export REDIS_PORT=6380
export INTENT_MODEL_PATH=./data/models
export CHROMA_PERSIST_DIR=./data/chroma
```

### Docker Environment

All environment variables are pre-configured in `docker-compose.test.yml`:

```yaml
environment:
  - AWS_ENDPOINT_URL=http://localstack:4566
  - CHROMADB_HOST=chromadb
  - REDIS_HOST=redis
  - TEST_MODE=true
```

## Requirements

### Local Testing
- Python 3.9+
- Docker & Docker Compose
- 8GB+ RAM

### Python Dependencies
```
pytest>=7.3.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-html>=3.2.0
faker>=18.0.0
moto>=4.1.0
httpx>=0.24.0
requests>=2.31.0
```

### Docker Services
- LocalStack (AWS simulation)
- ChromaDB (vector store)
- Redis (optional caching)

## Troubleshooting

### Tests Failing Due to Missing Data

```bash
# Regenerate test data
python tests/data/generate_synthetic_data.py

# Re-seed ChromaDB
python tests/scripts/seed_chroma.py
```

### LocalStack Connection Issues

```bash
# Restart LocalStack
docker-compose -f docker-compose.test.yml restart localstack

# Check LocalStack health
curl http://localhost:4566/_localstack/health
```

### ChromaDB Connection Issues

```bash
# Check ChromaDB health
curl http://localhost:8001/api/v1/heartbeat

# Re-seed if needed
python tests/scripts/seed_chroma.py
```

### Model Not Found

```bash
# The tests will use a mock model if ONNX model is not available
# To train a real model:
python -m src.intent_classifier.train --output-dir data/models
python -m src.intent_classifier.export_onnx --input-dir data/models --output-path data/models/intent_classifier.onnx
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose -f docker-compose.test.yml up -d localstack chromadb
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Generate test data
        run: python tests/data/generate_synthetic_data.py
      
      - name: Seed ChromaDB
        run: python tests/scripts/seed_chroma.py
      
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Performance Benchmarks

Expected performance on local CPU (M1 Mac / Intel i7):

| Component | Metric | Target | Typical |
|-----------|--------|--------|---------|
| Intent Classification | Latency | < 200ms | ~50-100ms |
| RAG Retrieval | Top-3 Accuracy | ≥ 80% | ~85-90% |
| RAG End-to-End | Latency | < 2s | ~500ms-1s |
| Feedback Analysis | Batch (100) | < 10s | ~3-5s |
| Topic Coherence | Score | > 0.4 | ~0.5-0.6 |

## Contributing

When adding new tests:

1. Place unit tests in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use descriptive test names
4. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.integration`)
5. Include docstrings explaining what the test validates
6. Ensure tests can run offline (no external API calls)

## License

MIT License - See LICENSE file for details
