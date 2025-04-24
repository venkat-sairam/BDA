FROM python:3.10-slim

RUN apt-get update && \
    apt-get clean

RUN apt-get install poppler-utils -y
RUN apt-get install tesseract-ocr -y

WORKDIR /

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
