FROM python:3.10-slim

WORKDIR /app

COPY app.py .
COPY models/ models/
COPY requirements-flask.txt .

RUN pip install --no-cache-dir -r requirements-flask.txt

EXPOSE 5000

CMD ["python", "app.py"]
