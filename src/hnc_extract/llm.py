"""LLM clients: an OpenAI-compatible HTTP client for vLLM, and offline mocks.

Per-model behaviour lives in a YAML `ModelConfig` (configs/models/*.yaml):
sampling, how reasoning effort is set, and whether JSON-schema constrained
decoding is requested. Constrained decoding is an optimization only: every
answer is parsed and validated by the caller regardless, with retries.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
import yaml
from pydantic import BaseModel, ConfigDict


class ReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_effort: str | None = None  # top-level request param (e.g. Qwen3.8)
    chat_template_kwargs: dict[str, Any] = {}  # e.g. {"enable_thinking": true}
    system_prompt_suffix: str | None = (
        None  # e.g. Muse Glimmer "Reasoning strength: high"
    )


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str  # short label used in run dirs and the review sheet
    provider: Literal["openai_compat", "mock"] = "openai_compat"
    served_model_name: str = "mock"
    base_url: str = "http://localhost:8000/v1"
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int = 16384
    # "auto": derive a per-call seed from the cache key (reproducible re-runs).
    seed: int | Literal["auto"] | None = "auto"
    structured_output: Literal["json_schema", "none"] = "json_schema"
    reasoning: ReasoningConfig = ReasoningConfig()
    extra_body: dict[str, Any] = {}
    max_attempts: int = 3  # validation/evidence retries per call
    request_timeout_s: float = 1800.0
    notes: str | None = None

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        cfg = cls.model_validate(yaml.safe_load(Path(path).read_text()))
        if url := os.environ.get("HNC_LLM_BASE_URL"):
            cfg = cfg.model_copy(update={"base_url": url})
        return cfg

    def fingerprint(self) -> str:
        """Hash of everything that can change the model's output."""
        relevant = self.model_dump(
            exclude={"name", "base_url", "request_timeout_s", "notes", "max_attempts"}
        )
        blob = json.dumps(relevant, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class Completion:
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0


class LLMClient(Protocol):
    def chat(
        self, messages: list[dict], schema: dict | None, seed: int | None, meta: dict
    ) -> Completion: ...


class OpenAICompatClient:
    """Chat completions against a vLLM (or any OpenAI-compatible) server."""

    def __init__(
        self,
        cfg: ModelConfig,
        transport_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        backoff_s: float = 2.0,
    ):
        self.cfg = cfg
        self.transport_retries = transport_retries
        self.backoff_s = backoff_s
        self.http = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=30.0),
            headers={
                "Authorization": f"Bearer {os.environ.get('HNC_LLM_API_KEY', 'EMPTY')}"
            },
            transport=transport,
        )

    def build_body(
        self, messages: list[dict], schema: dict | None, seed: int | None, name: str
    ) -> dict:
        c = self.cfg
        body: dict[str, Any] = {
            "model": c.served_model_name,
            "messages": messages,
            "temperature": c.temperature,
            "max_tokens": c.max_tokens,
        }
        for k in ("top_p", "top_k", "min_p"):
            if getattr(c, k) is not None:
                body[k] = getattr(c, k)
        if seed is not None:
            body["seed"] = seed
        if c.reasoning.reasoning_effort:
            body["reasoning_effort"] = c.reasoning.reasoning_effort
        if c.reasoning.chat_template_kwargs:
            body["chat_template_kwargs"] = c.reasoning.chat_template_kwargs
        if schema is not None and c.structured_output == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            }
        body.update(c.extra_body)
        return body

    def chat(
        self, messages: list[dict], schema: dict | None, seed: int | None, meta: dict
    ) -> Completion:
        body = self.build_body(messages, schema, seed, name=meta.get("field", "output"))
        delay = self.backoff_s
        for attempt in range(self.transport_retries + 1):
            t0 = time.monotonic()
            try:
                r = self.http.post("/chat/completions", json=body)
            except httpx.TransportError:
                if attempt == self.transport_retries:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code >= 500 or r.status_code == 429:
                if attempt == self.transport_retries:
                    r.raise_for_status()
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code >= 400:
                # e.g. prompt longer than max-model-len: not retryable, surface vLLM's message
                raise RuntimeError(f"HTTP {r.status_code} from server: {r.text[:1000]}")
            data = r.json()
            choice = data["choices"][0]
            msg = choice["message"]
            return Completion(
                content=msg.get("content") or "",
                reasoning=msg.get("reasoning") or msg.get("reasoning_content"),
                finish_reason=choice.get("finish_reason"),
                usage=data.get("usage") or {},
                latency_s=time.monotonic() - t0,
            )
        raise RuntimeError("unreachable")

    def list_models(self) -> list[str]:
        r = self.http.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]


Responder = Callable[[list[dict], dict | None, dict], str]


class MockClient:
    """Offline client: `responder(messages, schema, meta) -> content`."""

    def __init__(self, responder: Responder):
        self.responder = responder
        self.calls: list[dict] = []

    def chat(
        self, messages: list[dict], schema: dict | None, seed: int | None, meta: dict
    ) -> Completion:
        self.calls.append(
            {"messages": messages, "schema": schema, "seed": seed, "meta": meta}
        )
        return Completion(
            content=self.responder(messages, schema, meta), finish_reason="stop"
        )


class ScriptedClient(MockClient):
    """Returns queued responses in order (for unit tests of retry logic)."""

    def __init__(self, responses: list[str]):
        self._queue = list(responses)
        super().__init__(lambda m, s, meta: self._queue.pop(0))


def make_client(cfg: ModelConfig) -> LLMClient:
    if cfg.provider == "mock":
        from hnc_extract.mock import heuristic_responder

        return MockClient(heuristic_responder)
    return OpenAICompatClient(cfg)
