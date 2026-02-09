#!/usr/bin/env python3
"""CDK app entry point."""

import os
from aws_cdk import App, Environment
from stack import CustomerServiceAIStack

app = App()

# Get environment from context or use defaults
account = os.environ.get("CDK_DEFAULT_ACCOUNT", "123456789012")
region = os.environ.get("CDK_DEFAULT_REGION", "us-east-1")

CustomerServiceAIStack(
    app,
    "CustomerServiceAIStack",
    env=Environment(
        account=account,
        region=region,
    ),
)

app.synth()
