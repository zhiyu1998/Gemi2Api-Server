FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Build-time proxy support (pass via --build-arg if needed)
ARG HTTP_PROXY
ARG HTTPS_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}

# Install dependencies
COPY pyproject.toml .
RUN uv sync

# Copy application code
COPY main.py .
COPY assets/ assets/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]