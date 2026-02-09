FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ && \
    yum clean all

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code and model
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY model.py ${LAMBDA_TASK_ROOT}

# Download and cache model during build
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('distilbert-base-uncased')"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache
ENV HF_HOME=/tmp/huggingface

CMD ["lambda_function.handler"]
