# Use an official Python runtime as a parent image
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENV MODEL_NAME=fine_tuned_german_bert

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


