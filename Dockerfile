FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train_model.py"]
