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

# Upgrade pip and install prebuilt binary wheels (avoid compiling heavy deps from source)
RUN pip install --upgrade pip setuptools wheel
RUN pip config set global.extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt --only-binary=:all: --prefer-binary || pip install -r requirements.txt --no-cache-dir

# Copy the rest of the app code
COPY . .

# --- ✅ Add environment variable defaults (safe fallbacks for Render) ---
ENV PORT=5000
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# --- ✅ Optional: preload sentence-transformer model for faster startup ---
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose Flask/Gunicorn port
EXPOSE 5000

# --- ✅ Use Gunicorn to run Flask app in production ---
CMD ["gunicorn", "--workers", "2", "--threads", "4", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
