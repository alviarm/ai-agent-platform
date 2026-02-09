#!/bin/bash
# LocalStack Initialization Script
# Creates AWS resources for local testing

set -e

echo "=== Initializing LocalStack Resources ==="

# AWS Configuration
export AWS_ACCESS_KEY_ID=test
export AWS_SECRET_ACCESS_KEY=test
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION=us-east-1
export AWS_ENDPOINT_URL=http://localhost:4566

# Wait for LocalStack to be ready
echo "Waiting for LocalStack..."
until curl -s http://localhost:4566/_localstack/health > /dev/null; do
    sleep 2
done
echo "LocalStack is ready!"

# ===============================
# S3 Buckets
# ===============================
echo "Creating S3 buckets..."

# Bucket for ML models
aws --endpoint-url=$AWS_ENDPOINT_URL s3 mb s3://ai-agent-models \
    2>/dev/null || echo "Bucket ai-agent-models already exists"

# Bucket for conversation logs
aws --endpoint-url=$AWS_ENDPOINT_URL s3 mb s3://conversation-logs \
    2>/dev/null || echo "Bucket conversation-logs already exists"

# ===============================
# DynamoDB Tables
# ===============================
echo "Creating DynamoDB tables..."

# Conversations table
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb create-table \
    --table-name Conversations \
    --attribute-definitions AttributeName=conversation_id,AttributeType=S \
    --key-schema AttributeName=conversation_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    2>/dev/null || echo "Table Conversations already exists"

# Feedback table
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb create-table \
    --table-name Feedback \
    --attribute-definitions \
        AttributeName=feedback_id,AttributeType=S \
        AttributeName=conversation_id,AttributeType=S \
        AttributeName=timestamp,AttributeType=S \
    --key-schema AttributeName=feedback_id,KeyType=HASH \
    --global-secondary-indexes \
        'IndexName=ConversationIndex,KeySchema=[{AttributeName=conversation_id,KeyType=HASH},{AttributeName=timestamp,KeyType=RANGE}],Projection={ProjectionType=ALL}' \
    --billing-mode PAY_PER_REQUEST \
    2>/dev/null || echo "Table Feedback already exists"

# Prompt Versions table (for A/B testing)
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb create-table \
    --table-name PromptVersions \
    --attribute-definitions \
        AttributeName=version_id,AttributeType=S \
        AttributeName=is_active,AttributeType=S \
    --key-schema AttributeName=version_id,KeyType=HASH \
    --global-secondary-indexes \
        'IndexName=ActiveIndex,KeySchema=[{AttributeName=is_active,KeyType=HASH}],Projection={ProjectionType=ALL}' \
    --billing-mode PAY_PER_REQUEST \
    2>/dev/null || echo "Table PromptVersions already exists"

# Analytics Summary table
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb create-table \
    --table-name AnalyticsSummary \
    --attribute-definitions \
        AttributeName=metric_type,AttributeType=S \
        AttributeName=date,AttributeType=S \
    --key-schema \
        AttributeName=metric_type,KeyType=HASH \
        AttributeName=date,KeyType=RANGE \
    --billing-mode PAY_PER_REQUEST \
    2>/dev/null || echo "Table AnalyticsSummary already exists"

# ===============================
# CloudWatch Log Groups
# ===============================
echo "Creating CloudWatch log groups..."

aws --endpoint-url=$AWS_ENDPOINT_URL logs create-log-group \
    --log-group-name /csai/intent-classifier \
    2>/dev/null || echo "Log group intent-classifier already exists"

aws --endpoint-url=$AWS_ENDPOINT_URL logs create-log-group \
    --log-group-name /csai/response-generator \
    2>/dev/null || echo "Log group response-generator already exists"

aws --endpoint-url=$AWS_ENDPOINT_URL logs create-log-group \
    --log-group-name /csai/feedback-processor \
    2>/dev/null || echo "Log group feedback-processor already exists"

# ===============================
# Upload Dummy Model to S3
# ===============================
echo "Uploading dummy model files..."

# Create a dummy ONNX model file if it doesn't exist
MODEL_DIR="/data/models"
mkdir -p $MODEL_DIR

# Create a minimal ONNX model for testing
if [ ! -f "$MODEL_DIR/dummy_intent_classifier.onnx" ]; then
    python3 << 'EOF'
import numpy as np

# Create a simple ONNX model for testing
try:
    import onnx
    from onnx import numpy_helper, TensorProto
    from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
    
    # Input: token IDs (batch_size, seq_length)
    input_ids = make_tensor_value_info('input_ids', TensorProto.INT64, [1, 'seq_length'])
    attention_mask = make_tensor_value_info('attention_mask', TensorProto.INT64, [1, 'seq_length'])
    
    # Output: logits (batch_size, num_labels=7)
    output = make_tensor_value_info('output', TensorProto.FLOAT, [1, 7])
    
    # Create a simple MatMul node for demonstration
    # In reality, this would be a full BERT model
    weights = numpy_helper.from_array(
        np.random.randn(128, 7).astype(np.float32),
        name='weights'
    )
    
    bias = numpy_helper.from_array(
        np.random.randn(7).astype(np.float32),
        name='bias'
    )
    
    # Simplified model - just produces random logits
    # This is for testing infrastructure, not actual inference
    node1 = make_node('Constant', [], ['const_weights'], value=weights)
    node2 = make_node('Constant', [], ['const_bias'], value=bias)
    node3 = make_node('RandomNormalLike', ['input_ids'], ['random_input'], 
                      mean=0.0, scale=1.0, dtype=TensorProto.FLOAT)
    node4 = make_node('MatMul', ['random_input', 'const_weights'], ['matmul_out'])
    node5 = make_node('Add', ['matmul_out', 'const_bias'], ['output'])
    
    graph = make_graph(
        [node1, node2, node3, node4, node5],
        'intent_classifier',
        [input_ids, attention_mask],
        [output]
    )
    
    model = make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 13)])
    model.ir_version = 8
    
    onnx.save(model, '/data/models/dummy_intent_classifier.onnx')
    print("Created dummy ONNX model")
    
except ImportError:
    print("ONNX not available, creating placeholder file")
    with open('/data/models/dummy_intent_classifier.onnx', 'wb') as f:
        f.write(b'DUMMY_ONNX_MODEL')
EOF
fi

# Upload model to S3
aws --endpoint-url=$AWS_ENDPOINT_URL s3 cp \
    $MODEL_DIR/dummy_intent_classifier.onnx \
    s3://ai-agent-models/intent_classifier/dummy_model.onnx \
    2>/dev/null || echo "Model already uploaded"

# Create model metadata
if [ ! -f "$MODEL_DIR/model_metadata.json" ]; then
cat > $MODEL_DIR/model_metadata.json << 'EOF'
{
    "model_name": "distilbert-base-uncased",
    "task": "intent-classification",
    "num_labels": 7,
    "labels": ["return", "grievance", "billing", "technical", "support", "general_inquiry", "escalation"],
    "max_length": 128,
    "version": "1.0.0-test",
    "created_at": "2024-01-01T00:00:00Z"
}
EOF
fi

aws --endpoint-url=$AWS_ENDPOINT_URL s3 cp \
    $MODEL_DIR/model_metadata.json \
    s3://ai-agent-models/intent_classifier/metadata.json \
    2>/dev/null || echo "Metadata already uploaded"

# ===============================
# Seed DynamoDB with Test Data
# ===============================
echo "Seeding DynamoDB tables..."

# Add a sample prompt version
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb put-item \
    --table-name PromptVersions \
    --item '{
        "version_id": {"S": "v1.0.0"},
        "name": {"S": "Default Prompt"},
        "template": {"S": "You are a helpful customer service assistant. Answer the following question based on the context provided.\n\nContext: {context}\n\nConversation History: {history}\n\nUser: {query}\n\nAssistant:"},
        "is_active": {"S": "true"},
        "created_at": {"S": "2024-01-01T00:00:00Z"}
    }' \
    2>/dev/null || true

# ===============================
# Verification
# ===============================
echo ""
echo "=== Verification ==="
echo "S3 Buckets:"
aws --endpoint-url=$AWS_ENDPOINT_URL s3 ls

echo ""
echo "DynamoDB Tables:"
aws --endpoint-url=$AWS_ENDPOINT_URL dynamodb list-tables

echo ""
echo "CloudWatch Log Groups:"
aws --endpoint-url=$AWS_ENDPOINT_URL logs describe-log-groups --query 'logGroups[*].logGroupName' --output table

echo ""
echo "=== LocalStack Initialization Complete ==="
echo "All AWS resources are ready for local testing!"
