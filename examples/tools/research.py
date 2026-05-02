import logging
import os

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def web_search(query: str) -> str:
    """Search the web for current information.

    Args:
        query: The search query.

    Returns:
        Formatted search results as text.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return (
            "Web search is not configured. "
            "Set TAVILY_API_KEY in your .env file to enable this feature."
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=3)
        results = response.get("results", [])

        if not results:
            return f"No results found for '{query}'."

        lines = [f"Web search results for '{query}':"]
        for r in results:
            lines.append(f"\n**{r.get('title', 'No title')}**")
            lines.append(r.get("content", "")[:500])
            lines.append(f"Source: {r.get('url', '')}")

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"[web_search] Error: {e}")
        return f"Search failed: {e}"
