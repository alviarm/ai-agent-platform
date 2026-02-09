"""
Pytest Configuration and Shared Fixtures

This file contains shared fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Environment Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_env():
    """Set up test environment variables."""
    # Save original values
    original_env = {}
    test_vars = {
        "TEST_MODE": "true",
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_SECRET_ACCESS_KEY": "test",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_REGION": "us-east-1",
        "AWS_ENDPOINT_URL": "http://localhost:4566",
        "CHROMADB_HOST": "localhost",
        "CHROMADB_PORT": "8001",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6380",
        "USE_LOCALSTACK": "true",
        "INTENT_MODEL_PATH": str(project_root / "data" / "models"),
        "CHROMA_PERSIST_DIR": str(project_root / "data" / "chroma"),
    }
    
    # Set test values
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def project_root_dir():
    """Return path to project root directory."""
    return project_root


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def intent_classifier():
    """Provide an initialized intent classifier."""
    from src.intent_classifier import IntentClassifier
    
    model_path = project_root / "data" / "models"
    
    try:
        classifier = IntentClassifier(
            model_path=str(model_path) if model_path.exists() else None,
            use_onnx=False,
            device="cpu",
        )
        return classifier
    except Exception as e:
        pytest.skip(f"Could not initialize intent classifier: {e}")


@pytest.fixture(scope="session")
def vector_store():
    """Provide an initialized vector store."""
    from src.response_generator import VectorStore
    
    persist_dir = project_root / "data" / "chroma"
    
    try:
        store = VectorStore(persist_dir=str(persist_dir))
        return store
    except Exception as e:
        pytest.skip(f"Could not initialize vector store: {e}")


@pytest.fixture
def conversation_manager():
    """Provide a fresh conversation manager."""
    from src.response_generator import ConversationManager
    
    return ConversationManager(use_dynamodb=False)


@pytest.fixture
def prompt_manager():
    """Provide a prompt manager."""
    from src.response_generator import PromptManager
    
    return PromptManager()


@pytest.fixture
def feedback_analyzer():
    """Provide a feedback analyzer."""
    from src.feedback_pipeline import FeedbackAnalyzer
    
    try:
        # Use VADER for speed in tests
        return FeedbackAnalyzer(use_transformer_sentiment=False)
    except Exception as e:
        pytest.skip(f"Could not initialize feedback analyzer: {e}")


@pytest.fixture
def text_preprocessor():
    """Provide a text preprocessor."""
    from src.feedback_pipeline import TextPreprocessor
    
    return TextPreprocessor()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_intents_data(test_data_dir):
    """Load test intent classification data."""
    import json
    
    data_file = test_data_dir / "test_intents.json"
    
    if not data_file.exists():
        # Try to generate data
        generator_script = test_data_dir / "generate_synthetic_data.py"
        if generator_script.exists():
            import subprocess
            result = subprocess.run(
                [sys.executable, str(generator_script)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                pytest.skip(f"Could not generate test data: {result.stderr}")
    
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")
    
    with open(data_file) as f:
        data = json.load(f)
    
    return data.get("samples", [])


@pytest.fixture(scope="session")
def test_conversations_data(test_data_dir):
    """Load test conversation data."""
    import json
    
    data_file = test_data_dir / "test_conversations.jsonl"
    
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")
    
    conversations = []
    with open(data_file) as f:
        for line in f:
            conversations.append(json.loads(line))
    
    return conversations


@pytest.fixture(scope="session")
def test_feedback_data(test_data_dir):
    """Load test feedback data."""
    import csv
    
    data_file = test_data_dir / "test_feedback_raw.csv"
    
    if not data_file.exists():
        pytest.skip(f"Test data not found: {data_file}")
    
    feedbacks = []
    with open(data_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            feedbacks.append(dict(row))
    
    return feedbacks


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM response."""
    return "This is a mock response for testing purposes."


@pytest.fixture
def mock_litellm():
    """Mock litellm.completion for testing."""
    from unittest.mock import patch, MagicMock
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Mock LLM response"))
    ]
    
    with patch("litellm.completion", return_value=mock_response) as mock:
        yield mock


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def unique_id():
    """Generate a unique ID for tests."""
    import uuid
    return str(uuid.uuid4())


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        # Mark tests in integration folder as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Mark tests in unit folder as unit tests
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


def pytest_report_header(config):
    """Add custom header to test report."""
    return [
        "Customer Service AI Platform - Test Suite",
        f"Project Root: {project_root}",
    ]
