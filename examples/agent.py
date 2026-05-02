import logging
import os

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from examples.tools.notes import make_note_tools
from examples.tools.reminders import make_reminder_tools
from examples.tools.research import web_search

logger = logging.getLogger(__name__)

_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


SYSTEM_PROMPT = """You are MemoryOS — a personal AI assistant with persistent memory.

You remember facts about the user across all sessions using a Knowledge Graph.
You get smarter over time as you learn their preferences, projects, and relationships.

## Your Skill Domains

1. **Notes** — Create, search, tag, and organize personal notes
   Triggers: "remember this", "make a note", "save that", "write down"

2. **Reminders** — Set, list, and complete time-based reminders
   Triggers: "remind me", "don't let me forget", "set a reminder"

3. **Research** — Search the web for current information
   Triggers: "look up", "search for", "find out", "what is", "who is"

4. **Memory** — Explicitly store and recall facts about the user
   Triggers: "I'm working on", "I prefer", "remember that I", "what do you know about me"

## Skill Files
Skill details are in the /skills/ directory. Use read_file to load them when needed.
- Notes:     read_file("/notes/skill.yaml")
- Reminders: read_file("/reminders/skill.yaml")
- Research:  read_file("/research/skill.yaml")
- Memory:    read_file("/memory/skill.yaml")

## Guidelines
- Personalize every response using the memory context injected above (if present)
- When the user shares a new fact, use remember_fact to persist it immediately
- Proactively reference what you know about the user when relevant
- Be concise and helpful
"""


def get_chat_model(model_name: str = None):
    """Build a LangChain chat model based on CHAT_PROVIDER env var.

    Falls back to LLM_PROVIDER if CHAT_PROVIDER isn't set, so by default the
    chat model uses the same provider as memory extraction.

    Supported providers (each needs its own pip extra):
        openai     → langchain-openai
        groq       → langchain-groq
        anthropic  → langchain-anthropic
        gemini     → langchain-google-genai
        ollama     → langchain-ollama       (local)
        bedrock    → langchain-aws
    """
    provider = os.getenv("CHAT_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
    model = model_name or os.getenv("CHAT_MODEL", "")

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY") or _missing("GROQ_API_KEY", provider)
        model = model or "llama-3.3-70b-versatile"
        print(f"[Model] Using Groq chat model: {model}")
        return ChatGroq(model=model, api_key=api_key)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY") or _missing("ANTHROPIC_API_KEY", provider)
        model = model or "claude-3-5-sonnet-latest"
        print(f"[Model] Using Anthropic chat model: {model}")
        return ChatAnthropic(model=model, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or _missing("GEMINI_API_KEY", provider)
        model = model or "gemini-1.5-pro"
        print(f"[Model] Using Gemini chat model: {model}")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = model or "llama3.1:8b"
        print(f"[Model] Using Ollama chat model: {model} @ {base_url}")
        return ChatOllama(model=model, base_url=base_url)

    if provider == "bedrock":
        from langchain_aws import ChatBedrock
        model = model or "anthropic.claude-3-5-sonnet-20240620-v1:0"
        region = os.getenv("AWS_REGION", "us-east-1")
        print(f"[Model] Using Bedrock chat model: {model} ({region})")
        return ChatBedrock(model_id=model, region_name=region)

    # default — openai
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("OPENAI_API_KEY") or _missing("OPENAI_API_KEY", "openai")
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL") or None
    print(f"[Model] Using OpenAI chat model: {model}" + (f" @ {base_url}" if base_url else ""))
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url)


def _missing(var: str, provider: str):
    raise RuntimeError(f"{var} is required when CHAT_PROVIDER={provider}")


_agent = None


def create_memory_os_agent(memory, chat_model=None):
    """Build a singleton agent wired to a MemoryOS instance.

    Args:
        memory: MemoryOS instance (already initialized).
        chat_model: Optional pre-built LangChain BaseChatModel. If None,
                    `get_chat_model()` builds one from env vars. Pass your
                    own to use any provider not built in.
    """
    skills_backend = FilesystemBackend(
        root_dir=os.path.abspath(_SKILLS_DIR),
        virtual_mode=True,
    )

    driver_factory = memory.connection.get_driver
    note_tools = make_note_tools(driver_factory)
    reminder_tools = make_reminder_tools(driver_factory)

    all_tools = note_tools + reminder_tools + [web_search] + memory.tools

    agent = create_deep_agent(
        model=chat_model or get_chat_model(),
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        backend=skills_backend,
        middleware=[memory.middleware],
        checkpointer=memory.checkpointer,
        name="memory_os_agent",
    )

    logger.info(f"[MemoryOS] Agent created — {len(all_tools)} tools")
    print(f"[Agent] Created with {len(all_tools)} tools")
    return agent


def get_agent(memory=None, chat_model=None):
    global _agent
    if _agent is None:
        if memory is None:
            raise RuntimeError("get_agent() requires a MemoryOS instance on first call")
        _agent = create_memory_os_agent(memory, chat_model=chat_model)
    return _agent
