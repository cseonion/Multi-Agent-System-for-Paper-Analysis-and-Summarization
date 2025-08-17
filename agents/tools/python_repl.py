import logging
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# Try multiple import paths for compatibility across langchain versions
try:
    from langchain_experimental.tools.python.tool import PythonREPLTool  # type: ignore
    _HAS_PY_REPL = True
except Exception:
    try:
        from langchain_community.tools.python.tool import PythonREPLTool  # type: ignore
        _HAS_PY_REPL = True
    except Exception:
        PythonREPLTool = None  # type: ignore
        _HAS_PY_REPL = False


def get_python_repl_tool() -> Optional[BaseTool]:
    """
    Return a Python REPL tool for simple executable examples / pseudo-code sketches.
    The tool runs Python code in a restricted REPL (no internet). Use responsibly.
    Returns None if unavailable.
    """
    if not _HAS_PY_REPL:
        logger.warning("PythonREPLTool is unavailable (install langchain_experimental or langchain_community)")
        return None
    try:
        tool: BaseTool = PythonREPLTool()
        logger.info("Python REPL Tool enabled")
        return tool
    except Exception as e:
        logger.error(f"Failed to initialize PythonREPLTool: {e}")
        return None
