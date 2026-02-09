FROM public.ecr.aws/lambda/python:3.9

RUN yum update -y && yum install -y gcc gcc-c++ && yum clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY analyzer.py ${LAMBDA_TASK_ROOT}
COPY preprocessor.py ${LAMBDA_TASK_ROOT}

ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/tmp/nltk_data

CMD ["lambda_function.handler"]
