FROM python:3.12.4-slim-bookworm

# Copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock /app/

# Install dependencies using uv
RUN uv sync --locked

# Copy your app and model
COPY predict.py model.pkl /app/

# Expose port
EXPOSE 9696

# Launch FastAPI app
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]