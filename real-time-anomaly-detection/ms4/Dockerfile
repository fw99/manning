FROM python:3.7-slim

RUN mkdir service
COPY requirements.txt .
COPY main.py .
COPY model.joblib /service

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
