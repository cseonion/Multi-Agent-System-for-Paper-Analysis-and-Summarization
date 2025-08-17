import os
import logging
from typing import Optional

try:
    from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
    _HAS_TAVILY = True
except Exception:
    TavilySearchResults = None  # type: ignore
    _HAS_TAVILY = False

try:
    from langchain_community.tools import DuckDuckGoSearchResults  # type: ignore
    _HAS_DDG = True
except Exception:
    DuckDuckGoSearchResults = None  # type: ignore
    _HAS_DDG = False

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def get_web_search_tool(max_results: int = 5, prefer: Optional[str] = None) -> Optional[BaseTool]:
    """
    Create a reusable web search tool.

    Preference order:
    - If prefer == 'tavily' and Tavily is available with API key, use TavilySearchResults.
    - Else if prefer == 'ddg' and DuckDuckGo is available, use DuckDuckGoSearchResults.
    - Else try Tavily (with API key), then fallback to DuckDuckGo.

    Returns None if no web search tool is available.
    """
    tool = None

    def _make_tavily():
        if not _HAS_TAVILY:
            return None
        if not os.getenv("TAVILY_API_KEY"):
            logger.debug("TAVILY_API_KEY not set; skipping TavilySearchResults")
            return None
        try:
            return TavilySearchResults(max_results=max_results)  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to initialize TavilySearchResults: {e}")
            return None

    def _make_ddg():
        if not _HAS_DDG:
            return None
        try:
            return DuckDuckGoSearchResults()  # type: ignore
        except Exception as e:
            logger.warning(f"Failed to initialize DuckDuckGoSearchResults: {e}")
            return None

    if prefer == "tavily":
        tool = _make_tavily() or _make_ddg()
    elif prefer == "ddg":
        tool = _make_ddg() or _make_tavily()
    else:
        tool = _make_tavily() or _make_ddg()

    if tool is None:
        logger.warning("No web search tool available (Tavily/DDG missing)")
    else:
        logger.info(f"Web Search Tool enabled: {tool.__class__.__name__}")

    return tool
