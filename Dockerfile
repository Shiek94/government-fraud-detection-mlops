FROM public.ecr.aws/lambda/python:3.11

COPY requirements-api.txt ${LAMBDA_TASK_ROOT}/requirements-api.txt

RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-api.txt

COPY src/ ${LAMBDA_TASK_ROOT}/src/

ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src

CMD ["fraud_detection.api.handler"]