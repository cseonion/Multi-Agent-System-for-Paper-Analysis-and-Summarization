from typing import Optional
from langchain_core.tools import BaseTool, tool


def get_vectorstore_search_tool(vectorstore, paper_title: str, name: str = "vectorstore_search") -> Optional[BaseTool]:
    """
    Create a reusable vectorstore search tool for a specific paper.

    Args:
        vectorstore: A vectorstore instance exposing `.similarity_search(query, k)`.
        paper_title: Paper title (for error/context messages)
        name: Tool name (default: "vectorstore_search")

    Returns:
        A LangChain tool that searches the given vectorstore, or None if vectorstore is None.
    """
    if vectorstore is None:
        return None

    @tool(name)
    def _vectorstore_search(query: str, k: int = 5) -> str:
        """Search the paper's vectorstore with a natural-language query and return top-k chunks.
        Useful for verifying technical details, definitions, or methodology from the source document.
        Args:
            query: 검색할 자연어 질의
            k: 상위 반환 개수 (기본 5)
        """
        try:
            docs = vectorstore.similarity_search(query, k=k)
            if not docs:
                return "No results found in vectorstore."
            lines = []
            for i, d in enumerate(docs):
                meta = getattr(d, "metadata", {}) or {}
                src = meta.get("source") or meta.get("file_path") or meta.get("path")
                page = meta.get("page") or meta.get("page_number")
                prefix = f"[{i+1}] "
                if src and page is not None:
                    prefix += f"(source: {src}, page: {page}) "
                elif src:
                    prefix += f"(source: {src}) "
                lines.append(prefix + d.page_content)
            return "\n\n".join(lines)
        except Exception as e:
            return f"Vectorstore search error for '{paper_title}': {str(e)}"

    return _vectorstore_search
