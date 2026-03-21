# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. System dependencies (useful for scientific Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and setup files first
COPY requirements.txt setup.py ./
COPY src/__init__.py src/__init__.py

# 5. Install Python deps (including editable install)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the entire project into the image
COPY . .

# 6. Expose port (Render uses 10000)
EXPOSE 10000

# 7. Default command: run Flask app with gunicorn for production
CMD ["python", "-m", "gunicorn", "--bind=0.0.0.0:10000", "--workers=2", "--timeout=120", "application:app"]
