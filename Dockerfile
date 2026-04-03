FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev
COPY src/ ./src/
COPY amber-manifest.json5 ./
WORKDIR /app/src
ENV AGENT_PORT=9009
EXPOSE 9009
CMD ["uv", "run", "python", "server.py"]
