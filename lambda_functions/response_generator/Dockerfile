FROM public.ecr.aws/lambda/python:3.9

RUN yum update -y && yum install -y gcc gcc-c++ git && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY conversation_manager.py ${LAMBDA_TASK_ROOT}
COPY prompt_manager.py ${LAMBDA_TASK_ROOT}

ENV PYTHONUNBUFFERED=1
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/models

CMD ["lambda_function.handler"]
