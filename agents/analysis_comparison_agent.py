from src.state import State
from src.tracking import track_agent
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from agents.tools.web_search import get_web_search_tool
from agents.tools.vectorstore import get_vectorstore_search_tool
from config.agent_llm import get_llm

# 현재 모듈 로거 생성
logger = logging.getLogger(__name__)

instructions = """
You are a Comparison Analysis Agent specialized in comparing multiple papers within the same academic domain.
Your role is to synthesize and compare findings across papers for readers who may lack deep knowledge in the field.

# Key Requirements
1. Assume the reader has limited knowledge in at least one aspect of the domain
2. Explain domain-specific terminology when first introduced (brief, accessible definitions)
3. Use available tools (web search, vectorstore search) when helpful to verify methods, clarify definitions, or provide historical context
4. Separate technical details from interpretation
5. Present results in a clear, structured report

# Report Structure
- Executive Summary: High-level overview of similarities and differences
- Methodology Comparison (Table): Compare approaches across papers (columns example: Paper, Approach/Model, Data/Setup, Strengths, Weaknesses)
- Evolution Over Time: As an expert in the shared main field, discuss how approaches have evolved over time. The input may lack publication years; when needed, use your knowledge or web search to infer timeline and explain trends.
- Findings: Contradicting or Supporting Results across papers
- Shared Limitations: Common limitations or open challenges identified
- Synthesis and Recommendations: What to take away and suggested next steps

Tone: Clear, educational, and comparative. When using external info, integrate concisely and avoid over-citation.
"""

COMPARISON_LLM = get_llm("comparison_agent")


@track_agent("comparison_agent")
def comparison_agent(state: State) -> State:
    """동일 도메인 내 다중 논문 비교 분석 에이전트 (ReAct + Tools)"""
    paper_domains = state["paper_domain"]
    final_summaries = state["final_summary"]
    vectorstores = state.get("vectorstores", {})

    papers = list(final_summaries.keys())

    logger.info(f"🧮 비교 분석 시작: 대상 논문 수 {len(papers)}")

    # 공유 메인 필드 간소화: 첫 번째 논문의 main field 사용
    subfield_map = {}
    for p in papers:
        dom = paper_domains.get(p, {}) or {}
        mlist = dom.get("main_field", []) or []
        sflist = dom.get("sub_field", []) or []
        subfield_map[p] = {"main_field": mlist, "sub_field": sflist}
    shared_main_field = None
    if papers:
        first_main_fields = subfield_map[papers[0]]["main_field"]
        shared_main_field = first_main_fields[0] if first_main_fields else None

    # --- Tools 구성 ---
    tools = []

    # 1) Web Search Tool (공통 함수 사용)
    web_tool = get_web_search_tool(max_results=5)
    if web_tool is not None:
        tools.append(web_tool)
        logger.info("🔧 Web Search Tool 활성화")
    else:
        logger.warning("⚠️ Web Search Tool 비활성화 (사용 가능한 검색 툴 없음)")

    # 2) Vectorstore Search Tool (각 논문별로 주입, 이름을 구분)
    vs_tools_added = 0
    tool_name_map = []  # [(name, paper_title)]
    for idx, p in enumerate(papers, start=1):
        vs = vectorstores.get(p)
        tool_name = f"vectorstore_search_{idx}"
        vs_tool = get_vectorstore_search_tool(vs, p, name=tool_name)
        if vs_tool is not None:
            tools.append(vs_tool)
            vs_tools_added += 1
            tool_name_map.append((tool_name, p))
        else:
            logger.warning(f"⚠️ Vectorstore Search Tool 비활성화 (벡터스토어 없음: {p})")
    logger.info(f"🔧 Vectorstore Tools 추가됨: {vs_tools_added}개")

    # --- ReAct Agent 생성 ---
    agent = create_react_agent(COMPARISON_LLM, tools=tools, state_modifier=instructions)

    # 분석 프롬프트 생성 (사용자 메시지)
    summaries_block = []
    for i, p in enumerate(papers, start=1):
        summaries_block.append(
            f"Paper {i}: {p}\nMain Fields: {subfield_map[p]['main_field']}\nSub Fields: {subfield_map[p]['sub_field']}\n---\n{final_summaries[p]}\n"
        )
    summaries_text = "\n\n".join(summaries_block)

    tools_hint_lines = ["Tools available:"]
    if web_tool is not None:
        tools_hint_lines.append("- web_search: Use to check historical timelines or clarify general domain facts.")
    for name, p in tool_name_map:
        tools_hint_lines.append(f"- {name}: Vector search for '{p}'.")
    # tools_hints = "\n".join(tools_hint_lines)

    prompt = f"""
    You are an expert in the shared main field: {shared_main_field or 'the provided domain'}.
    Compare the following papers within the same domain and produce the report described in your instructions.

    # All Summaries
    {summaries_text}
    ---------------------------------------
    Follow the instruction and use tools (web search and vectorstore_search) when necessary to improve factual accuracy,
    verify definitions, and extract missing technical context from the source document. If the summary is sufficient, you may proceed without tools.
    """

    try:
        logger.info("🔍 ReAct 에이전트 실행 (tool 사용 가능, 다중 논문 비교)...")
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            analysis_result = messages[-1].content
        else:
            analysis_result = str(result)

        logger.info("✅ 비교 분석 완료")
        logger.debug(f"분석 결과 길이: {len(analysis_result)} 문자")

        # delta만 반환 + write 단계 호환을 위해 analysis_report도 세팅
        return {
            "analysis_result": analysis_result,
            "analysis_report": analysis_result,
        }

    except Exception as e:
        error_msg = f"비교 분석 중 오류 발생: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            "analysis_result": f"비교 분석 오류: {error_msg}",
            "analysis_report": f"비교 분석 오류: {error_msg}",
        }