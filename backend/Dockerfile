# Use a full Python image that supports scientific libraries
FROM python:3.11-slim

# Install system-level dependencies required for numpy, torch, spacy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Flask port
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["gunicorn", "app:app"]
