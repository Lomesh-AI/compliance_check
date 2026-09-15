FROM python:3.10-slim

# Install system dependencies for FAISS and PDFs
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the fixed application port
EXPOSE 8080

# Build missing indexes before starting the production WSGI server
CMD ["python", "scripts/start.py"]