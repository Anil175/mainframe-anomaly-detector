
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential wget git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY web /app/web

ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/model

RUN mkdir -p /app/data /app/model

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8080
CMD ["/app/start.sh"]
