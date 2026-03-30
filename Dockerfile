FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optimization: Better python logging
ENV PYTHONUNBUFFERED=1

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy the rest of the code before installing so pyproject.toml can find files
COPY . .

# Install dependencies and project using uv
RUN uv pip install --system .

# Port expose (HF Spaces standard)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:7860/health || exit 1

# Start the server using the entry point defined in pyproject.toml
# This ensures it passes the 'server' script requirement of OpenEnv
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
