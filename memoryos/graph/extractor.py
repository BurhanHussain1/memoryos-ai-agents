from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from memoryos.config import MemoryConfig
from memoryos.graph.schema import EdgeData, MemoryPayload, NodeData

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = """You are a personal memory extraction assistant.

Given a conversation between a user and an AI assistant, extract structured facts
about the USER ONLY (not the assistant). Output valid JSON matching this schema:

{
  "nodes": [
    {
      "type": "Fact|Project|Person|Preference|Note|Reminder",
      "key": {"<unique_field>": "<value>"},
      "properties": {"<field>": "<value>"}
    }
  ],
  "edges": [
    {
      "type": "KNOWS|WORKING_ON|KNOWS_PERSON|PREFERS|HAS_NOTE|HAS_REMINDER",
      "from_type": "User",
      "from_key": {"user_id": "<user_id>"},
      "to_type": "<NodeType>",
      "to_key": {"<unique_field>": "<value>"}
    }
  ]
}

Node key fields by type:
- Fact:       {"fact_id": "fact_<8chars>"}
- Project:    {"name": "<project_name>"}
- Person:     {"name": "<person_name>"}
- Preference: {"key": "<preference_key>"}
- Note:       {"note_id": "note_<8chars>"}
- Reminder:   {"reminder_id": "rem_<8chars>"}

Node property fields:
- Fact:       content, category (general/work/personal), confidence (0.0-1.0), created_at
- Project:    status (active/paused/completed), description
- Person:     relation, context
- Preference: value, confidence (0.0-1.0)
- Note:       title, content, tags (comma-separated), created_at
- Reminder:   title, due_date (ISO 8601), completed (false)

Rules:
- Only extract facts explicitly stated by the user
- created_at: use current ISO 8601 timestamp
- Return {"nodes": [], "edges": []} if nothing worth storing
- Output ONLY valid JSON, no markdown fences"""


def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _build_caller(config: MemoryConfig):
    """Returns (call_fn, provider_label).

    call_fn(system_prompt, user_prompt) -> str   (raw JSON text)

    The call_fn closes over the right SDK client/config so the extractor
    doesn't need to know which provider it's talking to.
    """
    # Highest priority — user-supplied function
    if callable(config.extraction_fn):
        return config.extraction_fn, "custom"

    provider = (config.llm_provider or "openai").lower()
    model = config.extraction_model
    temp = config.extraction_temperature
    max_toks = config.extraction_max_tokens
    extra = config.extraction_client_kwargs or {}

    # ── OpenAI / OpenAI-compatible ──────────────────────────────────────────
    if provider == "openai":
        from openai import OpenAI
        if not config.openai_api_key and not config.openai_base_url:
            raise ValueError("llm_provider='openai' requires openai_api_key or openai_base_url")
        client = OpenAI(
            api_key=config.openai_api_key or "not-needed",
            base_url=config.openai_base_url or None,
            **extra,
        )

        def call(system, user):
            r = client.chat.completions.create(
                model=model, temperature=temp, max_tokens=max_toks,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content
        return call, "openai"

    # ── Groq ────────────────────────────────────────────────────────────────
    if provider == "groq":
        from groq import Groq
        if not config.groq_api_key:
            raise ValueError("llm_provider='groq' requires groq_api_key (or GROQ_API_KEY env)")
        client = Groq(api_key=config.groq_api_key, **extra)

        def call(system, user):
            r = client.chat.completions.create(
                model=model, temperature=temp, max_tokens=max_toks,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return r.choices[0].message.content
        return call, "groq"

    # ── Anthropic (Claude) ─────────────────────────────────────────────────
    if provider == "anthropic":
        from anthropic import Anthropic
        if not config.anthropic_api_key:
            raise ValueError("llm_provider='anthropic' requires anthropic_api_key (or ANTHROPIC_API_KEY env)")
        client = Anthropic(api_key=config.anthropic_api_key, **extra)

        def call(system, user):
            r = client.messages.create(
                model=model, max_tokens=max_toks, temperature=temp,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return r.content[0].text if r.content else ""
        return call, "anthropic"

    # ── Gemini ──────────────────────────────────────────────────────────────
    if provider == "gemini":
        import google.generativeai as genai
        if not config.gemini_api_key:
            raise ValueError("llm_provider='gemini' requires gemini_api_key (or GEMINI_API_KEY env)")
        genai.configure(api_key=config.gemini_api_key)
        model_obj = genai.GenerativeModel(model)

        def call(system, user):
            r = model_obj.generate_content(
                f"{system}\n\n{user}",
                generation_config={"temperature": temp, "max_output_tokens": max_toks},
            )
            return r.text
        return call, "gemini"

    # ── Ollama (local) ─────────────────────────────────────────────────────
    if provider == "ollama":
        import requests
        base = config.ollama_base_url.rstrip("/")

        def call(system, user):
            r = requests.post(
                f"{base}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": max_toks},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        return call, "ollama"

    # ── LiteLLM (universal — 100+ providers) ───────────────────────────────
    if provider == "litellm":
        import litellm

        def call(system, user):
            r = litellm.completion(
                model=model,            # e.g. "anthropic/claude-3-5-sonnet", "bedrock/anthropic.claude-3", "vertex_ai/gemini-pro"
                temperature=temp,
                max_tokens=max_toks,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return r.choices[0].message.content
        return call, "litellm"

    raise ValueError(
        f"Unknown llm_provider: {provider!r}. "
        "Use one of: 'openai', 'groq', 'anthropic', 'gemini', 'ollama', 'litellm', 'custom'."
    )


class MemoryExtractor:
    def __init__(self, config: MemoryConfig):
        self._config = config
        self._call, self._provider = _build_caller(config)
        self._system_prompt = config.extraction_prompt or _DEFAULT_SYSTEM_PROMPT

    def extract(self, user_id: str, conversation: list[dict]) -> MemoryPayload:
        transcript = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation
        )
        now = datetime.now(UTC).isoformat()
        input_text = f"USER_ID: {user_id}\nCURRENT_TIME: {now}\n\nCONVERSATION:\n{transcript}"

        logger.info(
            f"[Extractor] Calling {self._provider}:{self._config.extraction_model} "
            f"for user={user_id} ({len(conversation)} msgs)"
        )

        try:
            raw = self._call(self._system_prompt, input_text) or ""
            data = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as e:
            logger.error(f"[Extractor] Invalid JSON: {e}")
            return MemoryPayload(user_id=user_id)
        except Exception as e:
            logger.error(f"[Extractor] Failed: {e}")
            return MemoryPayload(user_id=user_id)

        try:
            payload = MemoryPayload(
                user_id=user_id,
                nodes=[NodeData(**n) for n in data.get("nodes", [])],
                edges=[EdgeData(**e) for e in data.get("edges", [])],
            )
        except Exception as e:
            logger.error(f"[Extractor] Payload parse failed: {e}")
            return MemoryPayload(user_id=user_id)

        logger.info(f"[Extractor] nodes={len(payload.nodes)} edges={len(payload.edges)}")
        logger.info(f"[Extractor] Extracted {len(payload.nodes)} nodes, {len(payload.edges)} edges for user={user_id}")
        return payload
