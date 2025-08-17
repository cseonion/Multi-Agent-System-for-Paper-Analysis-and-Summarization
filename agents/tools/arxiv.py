import logging
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

try:
    from langchain_community.utilities import ArxivAPIWrapper  # type: ignore
    from langchain_community.tools import ArxivQueryRun  # type: ignore
    _HAS_ARXIV = True
except Exception:
    ArxivAPIWrapper = None  # type: ignore
    ArxivQueryRun = None  # type: ignore
    _HAS_ARXIV = False


def get_arxiv_search_tool(max_results: int = 5, sort_by: str = "relevance") -> Optional[BaseTool]:
    """
    Create a reusable arXiv search tool that returns metadata and abstracts only.

    Args:
        max_results: Maximum number of papers to retrieve
        sort_by: One of ["relevance", "lastUpdatedDate", "submittedDate"]

    Returns:
        A LangChain tool compatible with ReAct agents, or None if arXiv dependencies are missing.
    """
    if not _HAS_ARXIV:
        logger.warning("arXiv tool is unavailable (langchain_community/arxiv not installed)")
        return None

    try:
        wrapper = ArxivAPIWrapper(
            load_max_docs=max_results,
            sort_by=sort_by,
            doc_content_chars_max=4000,  # limit content; wrapper uses titles/abstracts by default
        )
        tool: BaseTool = ArxivQueryRun(api_wrapper=wrapper)
        logger.info("ArXiv Search Tool enabled: ArxivQueryRun")
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize ArXiv tool: {e}")
        return None
