# Pipeline runner: CPU only, talks HTTP to the vLLM service (network alias `llm`).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# specs/ and configs/ are bind-mounted by compose so edits need no rebuild.
ENTRYPOINT ["python", "-m", "hnc_extract"]
