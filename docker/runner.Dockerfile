# Pipeline runner: CPU only, talks HTTP to the vLLM service (network alias `llm`).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
# specs/ and configs/ are bind-mounted by compose so edits need no rebuild.
# umask 000: outputs are world-writable so the host user can edit/delete them.
ENTRYPOINT ["/bin/sh", "-c", "umask 000 && exec python -m hnc_extract \"$@\"", "hnc_extract"]
