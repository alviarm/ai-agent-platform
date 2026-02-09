"""AWS CDK Stack for Customer Service AI Platform."""

from constructs import Construct
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    Environment,
)
from aws_cdk import (
    aws_lambda as lambda_,
    aws_apigateway as apigw,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_iam as iam,
    aws_logs as logs,
    aws_cloudwatch as cloudwatch,
    aws_ecr as ecr,
)


class CustomerServiceAIStack(Stack):
    """CDK Stack for the Customer Service AI Platform."""
    
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env: Environment = None,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, env=env, **kwargs)
        
        # S3 Bucket for model artifacts and feedback uploads
        self.models_bucket = s3.Bucket(
            self,
            "ModelsBucket",
            bucket_name=f"csa-models-{self.account}",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        )
        
        self.feedback_bucket = s3.Bucket(
            self,
            "FeedbackBucket",
            bucket_name=f"csa-feedback-{self.account}",
            removal_policy=RemovalPolicy.RETAIN,
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ]
                )
            ],
        )
        
        # DynamoDB Tables
        self.conversations_table = dynamodb.Table(
            self,
            "ConversationsTable",
            table_name="csa-conversations",
            partition_key=dynamodb.Attribute(
                name="conversation_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
        )
        
        # GSI for querying by user_id
        self.conversations_table.add_global_secondary_index(
            index_name="user-id-index",
            partition_key=dynamodb.Attribute(
                name="user_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="updated_at",
                type=dynamodb.AttributeType.STRING,
            ),
        )
        
        self.feedback_table = dynamodb.Table(
            self,
            "FeedbackTable",
            table_name="csa-feedback",
            partition_key=dynamodb.Attribute(
                name="feedback_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )
        
        # GSI for querying by conversation_id
        self.feedback_table.add_global_secondary_index(
            index_name="conversation-index",
            partition_key=dynamodb.Attribute(
                name="conversation_id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
        )
        
        self.analytics_table = dynamodb.Table(
            self,
            "AnalyticsTable",
            table_name="csa-analytics",
            partition_key=dynamodb.Attribute(
                name="metric_name",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="timestamp",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
        )
        
        # ECR Repositories for Lambda container images
        self.intent_classifier_repo = ecr.Repository(
            self,
            "IntentClassifierRepo",
            repository_name="csa-intent-classifier",
            removal_policy=RemovalPolicy.RETAIN,
        )
        
        self.response_generator_repo = ecr.Repository(
            self,
            "ResponseGeneratorRepo",
            repository_name="csa-response-generator",
            removal_policy=RemovalPolicy.RETAIN,
        )
        
        self.feedback_processor_repo = ecr.Repository(
            self,
            "FeedbackProcessorRepo",
            repository_name="csa-feedback-processor",
            removal_policy=RemovalPolicy.RETAIN,
        )
        
        # Lambda Functions (using container images)
        # Note: Images need to be built and pushed to ECR separately
        
        # Intent Classification Lambda
        self.intent_lambda = lambda_.Function(
            self,
            "IntentClassifierFunction",
            function_name="csa-intent-classifier",
            description="Classify customer intent using BERT",
            code=lambda_.Code.from_ecr_image(
                repository=self.intent_classifier_repo,
                tag="latest",
            ),
            handler=lambda_.Handler.FROM_IMAGE,
            runtime=lambda_.Runtime.FROM_IMAGE,
            memory_size=3008,  # Max for ML inference
            timeout=Duration.seconds(30),
            environment={
                "MODEL_BUCKET": self.models_bucket.bucket_name,
                "MODEL_KEY": "models/intent_classifier.onnx",
                "LOG_LEVEL": "INFO",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )
        
        self.intent_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:ListBucket"],
                resources=[
                    self.models_bucket.bucket_arn,
                    f"{self.models_bucket.bucket_arn}/*",
                ],
            )
        )
        
        # Response Generator Lambda
        self.response_lambda = lambda_.Function(
            self,
            "ResponseGeneratorFunction",
            function_name="csa-response-generator",
            description="Generate responses using RAG",
            code=lambda_.Code.from_ecr_image(
                repository=self.response_generator_repo,
                tag="latest",
            ),
            handler=lambda_.Handler.FROM_IMAGE,
            runtime=lambda_.Runtime.FROM_IMAGE,
            memory_size=3008,
            timeout=Duration.seconds(60),
            environment={
                "MODEL_BUCKET": self.models_bucket.bucket_name,
                "CONVERSATIONS_TABLE": self.conversations_table.table_name,
                "OPENAI_API_KEY_SECRET": "csa/openai-api-key",
                "LOG_LEVEL": "INFO",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )
        
        self.response_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
                resources=[self.conversations_table.table_arn],
            )
        )
        
        self.response_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["secretsmanager:GetSecretValue"],
                resources=["arn:aws:secretsmanager:*:*:secret:csa/*"],
            )
        )
        
        # Feedback Processor Lambda
        self.feedback_lambda = lambda_.Function(
            self,
            "FeedbackProcessorFunction",
            function_name="csa-feedback-processor",
            description="Process feedback with NLP pipeline",
            code=lambda_.Code.from_ecr_image(
                repository=self.feedback_processor_repo,
                tag="latest",
            ),
            handler=lambda_.Handler.FROM_IMAGE,
            runtime=lambda_.Runtime.FROM_IMAGE,
            memory_size=2048,
            timeout=Duration.seconds(60),
            environment={
                "FEEDBACK_TABLE": self.feedback_table.table_name,
                "ANALYTICS_TABLE": self.analytics_table.table_name,
                "FEEDBACK_BUCKET": self.feedback_bucket.bucket_name,
                "LOG_LEVEL": "INFO",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )
        
        self.feedback_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:GetItem",
                    "dynamodb:PutItem",
                    "dynamodb:Query",
                ],
                resources=[
                    self.feedback_table.table_arn,
                    self.analytics_table.table_arn,
                    f"{self.feedback_table.table_arn}/index/*",
                ],
            )
        )
        
        self.feedback_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject", "s3:PutObject"],
                resources=[f"{self.feedback_bucket.bucket_arn}/*"],
            )
        )
        
        # API Gateway
        self.api = apigw.RestApi(
            self,
            "CustomerServiceAPI",
            rest_api_name="Customer Service AI API",
            description="API for Customer Service AI Platform",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                tracing_enabled=True,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
                metrics_enabled=True,
            ),
        )
        
        # Request validator
        request_validator = apigw.RequestValidator(
            self,
            "RequestValidator",
            rest_api=self.api,
            request_validator_name="csa-validator",
            validate_request_body=True,
            validate_request_parameters=True,
        )
        
        # /chat endpoint
        chat_resource = self.api.root.add_resource("chat")
        chat_integration = apigw.LambdaIntegration(
            self.response_lambda,
            proxy=True,
            integration_responses=[
                apigw.IntegrationResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": "'*'",
                    },
                ),
                apigw.IntegrationResponse(
                    status_code="500",
                    selection_pattern=".*[Internal Server Error].*",
                ),
            ],
        )
        
        chat_resource.add_method(
            "POST",
            chat_integration,
            request_validator=request_validator,
            method_responses=[
                apigw.MethodResponse(status_code="200"),
                apigw.MethodResponse(status_code="500"),
            ],
        )
        
        # /feedback endpoint
        feedback_resource = self.api.root.add_resource("feedback")
        feedback_integration = apigw.LambdaIntegration(
            self.feedback_lambda,
            proxy=True,
        )
        
        feedback_resource.add_method(
            "POST",
            feedback_integration,
            method_responses=[apigw.MethodResponse(status_code="200")],
        )
        
        feedback_resource.add_method(
            "GET",
            feedback_integration,
            method_responses=[apigw.MethodResponse(status_code="200")],
        )
        
        # /analytics endpoint
        analytics_resource = self.api.root.add_resource("analytics")
        analytics_integration = apigw.LambdaIntegration(
            self.feedback_lambda,
            proxy=True,
        )
        
        analytics_resource.add_method(
            "GET",
            analytics_integration,
            method_responses=[apigw.MethodResponse(status_code="200")],
        )
        
        # CloudWatch Alarms
        # High error rate alarm
        error_alarm = cloudwatch.Alarm(
            self,
            "HighErrorRateAlarm",
            metric=self.response_lambda.metric_errors(),
            threshold=10,
            evaluation_periods=2,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        
        # High latency alarm
        latency_alarm = cloudwatch.Alarm(
            self,
            "HighLatencyAlarm",
            metric=self.response_lambda.metric_duration(),
            threshold=5000,  # 5 seconds
            evaluation_periods=3,
            datapoints_to_alarm=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        
        # Outputs
        CfnOutput(self, "APIEndpoint", value=self.api.url)
        CfnOutput(self, "ModelsBucket", value=self.models_bucket.bucket_name)
        CfnOutput(self, "ConversationsTable", value=self.conversations_table.table_name)
        CfnOutput(self, "FeedbackTable", value=self.feedback_table.table_name)
