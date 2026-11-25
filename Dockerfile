FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN uv sync

# Copy application code
COPY *.py .
COPY templates/ templates/
COPY assets/ assets/

# Expose the port the app runs on
ENV PORT=7860
EXPOSE ${PORT}

# Command to run the application
CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
