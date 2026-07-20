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

# Expose the port Hugging Face expects
EXPOSE 7860

# Run the Flask app
CMD ["python", "src/api/app.py"]