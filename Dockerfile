FROM python:3.11.9-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Better python logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies directly via pip (no uv to keep build simple)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Port expose (HF Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:7860/health || exit 1

# Start the server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
