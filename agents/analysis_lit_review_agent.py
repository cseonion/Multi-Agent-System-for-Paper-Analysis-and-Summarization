from src.state import State
from src.tracking import track_agent
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from agents.tools.web_search import get_web_search_tool
from agents.tools.vectorstore import get_vectorstore_search_tool
from agents.tools.arxiv import get_arxiv_search_tool
from config.agent_llm import get_llm

# 현재 모듈 로거 생성
logger = logging.getLogger(__name__)

instructions = """
You are a Literature Review Analysis Agent specialized in synthesizing multiple papers that share a common theme across different domains.
Your role is to extract thematic insights, map the progression of research, and propose future research directions.

# Key Requirements
1. Assume the reader has limited knowledge in at least one domain
2. Briefly explain domain-specific terminology when first introduced
3. Use available tools (web search, vectorstore search, arXiv search) to verify concepts, clarify definitions, provide historical context, and scout recent related works (abstract-level only)
4. Separate technical descriptions from interpretation
5. Present results as a clear, structured report

# Report Structure
- Executive Summary: High-level overview of the shared theme and scope
- Common Themes Identified: Themes recurring across papers with concise evidence
- Research Progression Mapped: How approaches and understanding evolved over time (use web search if publication years are missing)
- Knowledge Gaps Highlighted: Limitations and open problems spanning the theme
- Future Research Directions Suggested: Actionable and realistic next steps (cite abstracts or general sources when helpful)

Tone: Clear, educational, and synthesis-focused. Integrate external info concisely and avoid over-citation.
"""

LIT_REVIEW_LLM = get_llm("lit_review_agent")

@track_agent("lit_review_agent")
def lit_review_agent(state: State) -> State:
    """문헌 리뷰 에이전트 (ReAct + Tools, 공통 주제 / 상이한 도메인)"""
    paper_domains = state["paper_domain"]
    final_summaries = state["final_summary"]
    vectorstores = state.get("vectorstores", {})

    papers = list(final_summaries.keys())
    logger.info(f"📚 Literature Review 시작: 대상 논문 수 {len(papers)}")

    # 각 논문별 도메인 정보 수집 (표시용)
    subfield_map = {}
    for p in papers:
        dom = paper_domains.get(p, {}) or {}
        mlist = dom.get("main_field", []) or []
        sflist = dom.get("sub_field", []) or []
        subfield_map[p] = {"main_field": mlist, "sub_field": sflist}

    # --- Tools 구성 ---
    tools = []

    # 1) Web Search Tool (공통 함수 사용)
    web_tool = get_web_search_tool(max_results=5)
    if web_tool is not None:
        tools.append(web_tool)
        logger.info("🔧 Web Search Tool 활성화")
    else:
        logger.warning("⚠️ Web Search Tool 비활성화 (사용 가능한 검색 툴 없음)")

    # 2) Vectorstore Search Tool (각 논문별로 주입, 이름 구분)
    tool_name_map = []  # [(name, paper_title)]
    for idx, p in enumerate(papers, start=1):
        vs = vectorstores.get(p)
        tool_name = f"vectorstore_search_{idx}"
        vs_tool = get_vectorstore_search_tool(vs, p, name=tool_name)
        if vs_tool is not None:
            tools.append(vs_tool)
            tool_name_map.append((tool_name, p))
        else:
            logger.warning(f"⚠️ Vectorstore Search Tool 비활성화 (벡터스토어 없음: {p})")

    # 3) arXiv Search Tool (초록 수준 정보만 탐색)
    arxiv_tool = get_arxiv_search_tool(max_results=5, sort_by="relevance")
    if arxiv_tool is not None:
        tools.append(arxiv_tool)
        logger.info("🔧 arXiv Search Tool 활성화")
    else:
        logger.warning("⚠️ arXiv Search Tool 비활성화 (환경 또는 의존성 문제)")

    # --- ReAct Agent 생성 ---
    agent = create_react_agent(LIT_REVIEW_LLM, tools=tools, state_modifier=instructions)

    # 분석 프롬프트 생성 (사용자 메시지)
    summaries_block = []
    for i, p in enumerate(papers, start=1):
        summaries_block.append(
            f"Paper {i}: {p}\nMain Fields: {subfield_map[p]['main_field']}\nSub Fields: {subfield_map[p]['sub_field']}\n---\n{final_summaries[p]}\n"
        )
    summaries_text = "\n\n".join(summaries_block)

    tools_hint_lines = ["Tools available:"]
    if web_tool is not None:
        tools_hint_lines.append("- web_search: Use to infer timelines and clarify domain concepts.")
    for name, p in tool_name_map:
        tools_hint_lines.append(f"- {name}: Vector search for '{p}'.")
    if arxiv_tool is not None:
        tools_hint_lines.append("- arxiv: Search for recent related works (abstract-level only).")
    tools_hints = "\n".join(tools_hint_lines)

    prompt = f"""
    Synthesize a literature review across papers that share a common theme but span different domains.

    # All Summaries
    {summaries_text}
    ---------------------------------------
    Follow the instruction and use tools (web search and vectorstore_search) when necessary to improve factual accuracy,
    verify definitions, and extract missing technical context from the source document. If the summary is sufficient, you may proceed without tools.
    *Caution: Do NOT include future research directions (handled by a separate ideation agent).
    """

    try:
        logger.info("🔍 ReAct 에이전트 실행 (tool 사용 가능, 문헌 리뷰)...")
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            analysis_report = messages[-1].content
        else:
            analysis_report = str(result)

        logger.info("✅ Literature Review 완료")
        logger.debug(f"분석 결과 길이: {len(analysis_report)} 문자")

        return {
            **state,
            "analysis_report": analysis_report
        }

    except Exception as e:
        error_msg = f"문헌 리뷰 중 오류 발생: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            **state,
            "analysis_report": f"문헌 리뷰 오류: {error_msg}"
        }