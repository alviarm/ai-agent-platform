.PHONY: help install train test api docker-build deploy clean

help: ## Show this help message
	@echo "Customer Service AI Platform - Available Commands"
	@echo "================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

dev-setup: ## Setup development environment
	pip install -r requirements.txt
	pip install pytest black isort mypy pre-commit
	pre-commit install

train: ## Train the intent classification model
	python scripts/train_intent_classifier.py

test: ## Run all tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	pytest tests/integration/ -v -s

test-e2e: ## Run end-to-end tests (requires services running)
	pytest tests/integration/test_e2e.py -v -s

test-quick: ## Run quick tests (skip slow ones)
	pytest tests/ -v -m "not slow"

# Local Testing Infrastructure
test-infra-up: ## Start local testing infrastructure (LocalStack, ChromaDB, Redis)
	docker-compose -f docker-compose.test.yml up -d localstack chromadb redis
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Services should be ready. Check with: make test-infra-status"

test-infra-down: ## Stop local testing infrastructure
	docker-compose -f docker-compose.test.yml down

test-infra-status: ## Check status of local testing infrastructure
	@docker-compose -f docker-compose.test.yml ps
	@echo ""
	@echo "Checking LocalStack health..."
	@curl -s http://localhost:4566/_localstack/health || echo "LocalStack not responding"
	@echo ""
	@echo "Checking ChromaDB health..."
	@curl -s http://localhost:8001/api/v1/heartbeat || echo "ChromaDB not responding"

test-data-generate: ## Generate synthetic test data
	python tests/data/generate_synthetic_data.py

test-chroma-seed: ## Seed ChromaDB with FAQ documents
	python tests/scripts/seed_chroma.py

test-docker: ## Run full test suite in Docker
	docker-compose -f docker-compose.test.yml up test-runner

test-docker-api: ## Start API server in Docker for manual testing
	docker-compose -f docker-compose.test.yml --profile api up api-server

test-setup: test-infra-up test-data-generate test-chroma-seed ## Full test setup
	@echo "Test environment ready!"

api: ## Start the API server locally
	python -m src.api.main

lint: ## Run linting
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	flake8 src/ tests/ scripts/

type-check: ## Run type checking
	mypy src/

load-faq: ## Load FAQ documents into vector store
	python -c "from src.response_generator import VectorStore; vs = VectorStore(); vs.load_faq_documents('data/faq_documents.json'); print('FAQ documents loaded')"

docker-build-intent: ## Build intent classifier Lambda container
	docker build -t csa-intent-classifier:latest lambda_functions/intent_classifier/

docker-build-response: ## Build response generator Lambda container
	docker build -t csa-response-generator:latest lambda_functions/response_generator/

docker-build-feedback: ## Build feedback processor Lambda container
	docker build -t csa-feedback-processor:latest lambda_functions/feedback_processor/

docker-build-all: docker-build-intent docker-build-response docker-build-feedback ## Build all Lambda containers

cdk-bootstrap: ## Bootstrap CDK
	cd src/infrastructure && cdk bootstrap

cdk-deploy: ## Deploy CDK stack
	cd src/infrastructure && cdk deploy

cdk-destroy: ## Destroy CDK stack
	cd src/infrastructure && cdk destroy

sam-build: ## Build SAM application
	sam build

sam-local-api: ## Run SAM local API
	sam local start-api

sam-deploy: ## Deploy SAM application
	sam deploy --guided

upload-model: ## Upload model to S3
	aws s3 cp data/models/onnx/model.onnx s3://csa-models-$$(aws sts get-caller-identity --query Account --output text)/models/intent_classifier.onnx

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/

benchmark: ## Benchmark ONNX vs PyTorch
	python -c "from src.intent_classifier.export_onnx import benchmark_onnx; benchmark_onnx('data/models/onnx', 'data/models/pytorch_model', num_samples=1000)"

evaluate: ## Evaluate intent classifier
	python -m src.intent_classifier.evaluate --model-path data/models/pytorch_model --output-dir data/models/evaluation

all: install train test ## Install, train, and test
