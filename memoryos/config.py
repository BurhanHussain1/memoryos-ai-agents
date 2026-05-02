from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryConfig:
    # ── Graph backend ────────────────────────────────────────────────────
    graph_backend: str = "neo4j"            # "neo4j" or "memgraph"
    graph_url: str = "bolt://localhost:7687"
    graph_username: str = "neo4j"
    graph_password: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""

    # ── Memory extraction LLM ─────────────────────────────────────────────
    # Provider options:
    #   "openai"     — OpenAI SDK
    #   "groq"       — Groq SDK
    #   "anthropic"  — Anthropic SDK (Claude)
    #   "gemini"     — google-generativeai (Gemini)
    #   "ollama"     — local Ollama server
    #   "litellm"    — universal proxy (100+ providers via "provider/model" strings)
    #   "custom"     — user-supplied extraction_fn
    llm_provider: str = "openai"

    # API keys — set the one(s) for your chosen provider
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""

    # Endpoint overrides (for local / self-hosted)
    openai_base_url: str = ""               # e.g. "http://localhost:11434/v1" for Ollama compat
    ollama_base_url: str = "http://localhost:11434"

    # Model name — interpreted by the active provider
    extraction_model: str = "gpt-4o-mini"

    # Custom extraction function — overrides everything else.
    # Signature: (system_prompt: str, user_prompt: str) -> str (returns raw JSON).
    # Use this for Bedrock, Vertex, Cohere, fine-tuned models, or any provider
    # not built in. Highest priority — when set, llm_provider is ignored.
    extraction_fn: Callable[[str, str], str] | None = None

    # Tuning
    extraction_prompt: str = ""             # optional override of default system prompt
    extraction_temperature: float = 0.1
    extraction_max_tokens: int = 2048

    # Extra kwargs passed to the SDK client constructor (e.g. timeout, organization)
    extraction_client_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── STM tuning ────────────────────────────────────────────────────────
    max_stm_messages: int = 200
    stm_ttl_seconds: int = 86400           # 24 h
    context_cache_ttl: int = 300           # 5 min

    @classmethod
    def from_env(cls) -> MemoryConfig:
        """Build config from environment variables (reads .env if present)."""
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        backend = os.getenv("GRAPH_BACKEND", "neo4j").lower()
        if backend == "memgraph":
            graph_url = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
            graph_user = os.getenv("MEMGRAPH_USERNAME", "")
            graph_pass = os.getenv("MEMGRAPH_PASSWORD", "")
        else:
            graph_url = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            graph_user = os.getenv("NEO4J_USERNAME", "neo4j")
            graph_pass = os.getenv("NEO4J_PASSWORD", "")

        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        provider_defaults = {
            "openai":    "gpt-4o-mini",
            "groq":      "llama-3.3-70b-versatile",
            "anthropic": "claude-3-5-haiku-20241022",
            "gemini":    "gemini-1.5-flash",
            "ollama":    "llama3.1:8b",
            "litellm":   "openai/gpt-4o-mini",
        }
        default_model = provider_defaults.get(provider, "gpt-4o-mini")

        return cls(
            graph_backend=backend,
            graph_url=graph_url,
            graph_username=graph_user,
            graph_password=graph_pass,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_password=os.getenv("REDIS_PASSWORD", ""),
            llm_provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            openai_base_url=os.getenv("OPENAI_BASE_URL", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            extraction_model=os.getenv("MEMORY_EXTRACTION_MODEL", default_model),
        )
