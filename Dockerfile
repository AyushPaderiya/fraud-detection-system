# 1. Base image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. System dependencies (useful for scientific Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the image
COPY . .

# 6. Expose Flask port
EXPOSE 5000

# 7. Default command: run Flask app
CMD ["python", "application.py"]
