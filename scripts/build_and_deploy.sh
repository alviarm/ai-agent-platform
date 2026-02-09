#!/bin/bash
# Build and deploy script for AWS Lambda containers

set -e

AWS_REGION=${AWS_REGION:-"us-east-1"}
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "========================================"
echo "Building and Deploying Lambda Containers"
echo "AWS Region: ${AWS_REGION}"
echo "ECR Registry: ${ECR_REGISTRY}"
echo "========================================"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Function to build and push
deploy_function() {
    local func_name=$1
    local dir_path=$2
    local ecr_repo="csa-${func_name}"
    
    echo ""
    echo "----------------------------------------"
    echo "Deploying: ${func_name}"
    echo "----------------------------------------"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names ${ecr_repo} --region ${AWS_REGION} 2>/dev/null || \
        aws ecr create-repository --repository-name ${ecr_repo} --region ${AWS_REGION}
    
    # Build image
    echo "Building Docker image..."
    docker build -t ${ecr_repo}:latest ${dir_path}
    
    # Tag image
    docker tag ${ecr_repo}:latest ${ECR_REGISTRY}/${ecr_repo}:latest
    
    # Push image
    echo "Pushing to ECR..."
    docker push ${ECR_REGISTRY}/${ecr_repo}:latest
    
    echo "✓ ${func_name} deployed successfully"
}

# Deploy Intent Classifier
deploy_function "intent-classifier" "lambda_functions/intent_classifier"

# Deploy Response Generator
deploy_function "response-generator" "lambda_functions/response_generator"

# Deploy Feedback Processor
deploy_function "feedback-processor" "lambda_functions/feedback_processor"

echo ""
echo "========================================"
echo "All Lambda functions deployed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Deploy CDK stack: cd src/infrastructure && cdk deploy"
echo "2. Upload model to S3: aws s3 cp data/models/onnx/model.onnx s3://csa-models-${ACCOUNT_ID}/models/"
echo "3. Test the API using the provided API Gateway URL"
