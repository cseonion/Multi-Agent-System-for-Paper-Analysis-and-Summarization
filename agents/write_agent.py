from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from src.state import State
from src.tracking import track_agent
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from agents.tools.python_repl import get_python_repl_tool

logger = logging.getLogger(__name__)

instructions = """
You are a technical blog writer (Medium/Tistory style) who turns analysis into an engaging article for a broad technical audience.

Output format: Markdown only. Always use proper Markdown structure with #, ##, ### headings, bullet lists, links, and fenced code blocks (```python ... ``` for code).

Style & Requirements
- Tone: clear, informative, slightly narrative; avoid academic report tone
- Audience: engineers and researchers across domains
- Structure: catchy title, short intro, main sections with headings, optional code blocks, conclusion, and bullet-point takeaways
- Use examples or toy snippets when helpful; keep them concise
- When a concept is implementable, you may sketch pseudo-code using Python (via a Python REPL tool) to demonstrate how one might implement it. Keep code safe and self-contained.
- Avoid hallucinating exact dataset names or results; if unsure, state assumptions
"""

WRITER_LLM = ChatOpenAI(
    model="gpt-5",
)


@track_agent("write_agent")
def write_agent(state: State) -> State:
    """블로그 스타일 라이팅 에이전트. 
    - analysis_plan == 'single'이면 요약본 기반 작성
    - 그 외(cross_domain, comparison, literature_review)는 분석 결과 기반 작성
    - 필요한 경우 PythonREPLTool로 간단한 의사 코드/예제 포함
    """
    analysis_plan = state.get("analysis_plan")
    summaries = state.get("final_summary", {})
    analysis_report = state.get("analysis_report")
    cache_dir = state.get("cache_dir")

    # 소스 선택 및 파일명 베이스 결정
    if analysis_plan == "single":
        if not summaries:
            return {**state, "final_report": "요약본이 없어 글을 생성할 수 없습니다."}
        paper_title = next(iter(summaries.keys()))
        source_text = summaries[paper_title]
        context_heading = f"Based on the paper summary: {paper_title}"
        base_name = paper_title
    else:
        if not analysis_report:
            return {**state, "final_report": "분석 결과가 없어 글을 생성할 수 없습니다."}
        source_text = analysis_report
        context_heading = "Based on the multi-paper analysis report"
        base_name = f"{analysis_plan or 'analysis'}"

    # 도구 구성 (Python REPL)
    tools = []
    py_tool = get_python_repl_tool()
    if py_tool is not None:
        tools.append(py_tool)
        logger.info("🔧 Python REPL Tool 활성화")

    agent = create_react_agent(WRITER_LLM, tools=tools, state_modifier=instructions)

    prompt = f"""
    Write a Medium/Tistory-style technical blog post in valid Markdown.

    Context Heading: {context_heading}

    Source Text
    -----------------
    {source_text}
    -----------------

    Requirements
    1) Start with a compelling, descriptive title (use a top-level # heading)
    2) Provide a short introduction with motivation and what readers will learn
    3) Organize the body with clear headings (#, ##, ###) and concise explanations
    4) If there are concepts that can be implemented in code, include a small Python pseudo-code or runnable snippet in a fenced block (```python)
    5) Conclude with key takeaways (bullet list) and potential applications
    6) Keep the tone approachable and practical; avoid dry academic writing
    """

    try:
        logger.info("📝 Write agent 실행 (블로그 스타일 생성, Markdown 출력)...")
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            blog_post = messages[-1].content
        else:
            blog_post = str(result)

        logger.info("✅ 블로그 포스트 생성 완료")
        return {**state, "final_report": blog_post}

    except Exception as e:
        logger.error(f"❌ 글 생성 중 오류: {e}")
        return {**state, "blog_post": f"글 생성 오류: {e}"}