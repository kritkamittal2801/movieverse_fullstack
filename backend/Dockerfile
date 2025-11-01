# Use a stable Python image that supports scientific libraries
FROM python:3.11-slim

# Install system-level dependencies required for numpy, torch, spacy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    curl \
    git \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list and install dependencies
COPY requirements.txt .

# Upgrade pip and install prebuilt binary wheels (avoid compiling blis from source)
RUN pip install --upgrade pip setuptools wheel
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --only-binary=:all: --prefer-binary || pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Start the Flask app with Gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
