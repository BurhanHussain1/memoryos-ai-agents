import os
import sys
import socket

from dotenv import load_dotenv

load_dotenv()


def check_env():
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    provider_key = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
    required = [provider_key, "NEO4J_PASSWORD"]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        print(f"ERROR: Missing required env vars: {missing}")
        print(f"(LLM_PROVIDER={provider} — make sure {provider_key} is set in .env)")
        sys.exit(1)
    print(f"Using LLM provider: {provider}")


def check_tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def preflight():
    check_env()
    print("Running preflight checks...")

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    redis_ok = check_tcp(redis_host, redis_port)
    neo4j_ok = check_tcp("localhost", 7687)

    print(f"  Redis:  {'OK' if redis_ok else 'UNAVAILABLE (checkpointing will use in-memory)'} ({redis_host}:{redis_port})")
    print(f"  Neo4j:  {'OK' if neo4j_ok else 'FAIL'} (localhost:7687)")

    if not neo4j_ok:
        print("\nNeo4j is not reachable. Start it with:")
        print("  docker compose up -d neo4j")
        sys.exit(1)

    print("Preflight passed.\n")


if __name__ == "__main__":
    preflight()

    import uvicorn

    uvicorn.run(
        "examples.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("API_RELOAD", "true").lower() == "true",
    )
