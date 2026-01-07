# Base image
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libffi-dev \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Hugging Face mirror
ENV HF_ENDPOINT=https://hf-mirror.com
# ENV HF_ENDPOINT=https://huggingface.co
# Install Python packages (use Aliyun mirror)
COPY docker/requirements.txt .
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/ && \
    pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# Copy application code
COPY docker/ ./docker/
COPY src/ ./src/

# Set Python import path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8000

# Start the docker
CMD ["uvicorn", "memos.api.server_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
