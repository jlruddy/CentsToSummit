FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

# Copy source and install project
COPY src/ ./src/
COPY README.md ./
RUN uv sync --no-dev

RUN mkdir -p /app/data /root/.kalshi && chmod 700 /root/.kalshi

ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8000
EXPOSE 8000

CMD ["uv", "run", "kalshi-poc", "run", "--host", "0.0.0.0", "--port", "8000"]
