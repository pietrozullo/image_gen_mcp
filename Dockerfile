FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

EXPOSE 3000

CMD ["python", "server.py"]
