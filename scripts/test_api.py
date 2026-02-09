#!/usr/bin/env python3
"""Test script for the API endpoints."""

import json
import sys
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_classify():
    """Test intent classification."""
    print("\nTesting /classify...")
    test_cases = [
        {"text": "I want to return my order"},
        {"text": "I'm very disappointed with your service"},
        {"text": "The app keeps crashing"},
    ]
    
    for test in test_cases:
        response = requests.post(f"{BASE_URL}/classify", json=test)
        result = response.json()
        print(f"\nInput: {test['text']}")
        print(f"Predicted intent: {result.get('intent')} (confidence: {result.get('confidence'):.3f})")


def test_chat():
    """Test chat endpoint."""
    print("\nTesting /chat...")
    
    # First message
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "I want to return my order",
        "user_id": "test_user_123",
    })
    result = response.json()
    print(f"\nUser: I want to return my order")
    print(f"Assistant: {result.get('response')}")
    print(f"Intent: {result.get('intent')} (confidence: {result.get('confidence'):.3f})")
    
    conversation_id = result.get("conversation_id")
    
    # Follow-up
    response = requests.post(f"{BASE_URL}/chat", json={
        "message": "It's order #12345",
        "conversation_id": conversation_id,
    })
    result = response.json()
    print(f"\nUser: It's order #12345")
    print(f"Assistant: {result.get('response')}")


def test_feedback():
    """Test feedback endpoint."""
    print("\nTesting /feedback...")
    
    test_feedbacks = [
        {"conversation_id": "test-123", "text": "The response was very helpful, thank you!", "rating": "positive"},
        {"conversation_id": "test-123", "text": "This didn't solve my problem at all", "rating": "negative"},
    ]
    
    for feedback in test_feedbacks:
        response = requests.post(f"{BASE_URL}/feedback", json=feedback)
        result = response.json()
        print(f"\nFeedback: {feedback['text']}")
        print(f"Analysis: sentiment={result.get('sentiment')}, confidence={result.get('confidence'):.3f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Customer Service AI API Tests")
    print("=" * 60)
    
    try:
        if not test_health():
            print("Health check failed! Make sure the API is running.")
            sys.exit(1)
        
        test_classify()
        test_chat()
        test_feedback()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to {BASE_URL}")
        print("Make sure the API server is running: python -m src.api.main")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
