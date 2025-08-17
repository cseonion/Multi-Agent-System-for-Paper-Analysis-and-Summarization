from src.state import State
from src.tracking import track_agent
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import os
from agents.tools.web_search import get_web_search_tool
from agents.tools.vectorstore import get_vectorstore_search_tool

# 현재 모듈 로거 생성
logger = logging.getLogger(__name__)

instructions = """
You are a Cross-Domain Analysis Agent specialized in analyzing papers that span multiple academic disciplines.

Your role is to create comprehensive analysis reports for readers who may lack knowledge in one or more of the domains covered by the paper. 

# Key Requirements
1. ASSUME the reader has limited knowledge in at least one domain
2. EXPLAIN all domain-specific terminology with clear, accessible definitions
3. IDENTIFY cross-field implications and connections between different domains
4. SEPARATE technical results from domain-specific interpretations
5. HIGHLIGHT interdisciplinary research opportunities and future directions
6. FORMAT the output as a structured research report

# Report Structure
- Executive Summary: High-level overview accessible to general academic audience
- Domain Analysis: Breakdown of each field involved with terminology explanations
- Cross-Domain Connections: How different fields interact in this research
- Technical Findings: Clear separation of methodology and results by domain
- Interdisciplinary Impact: Implications across all relevant fields
- Future Opportunities: Potential research directions at domain intersections

Write in a clear, educational tone that bridges knowledge gaps between disciplines.
"""

CROSS_DOMAIN_LLM = ChatOpenAI(
    model="gpt-5",
)


@track_agent("cross_domain_agent")
def cross_domain_agent(state: State) -> State:
    """교차 도메인 분석 에이전트 (ReAct + Tools)"""
    paper_domains = state["paper_domain"]
    final_summaries = state["final_summary"]
    vectorstores = state.get("vectorstores", {})

    # 단일 논문에 대한 교차 도메인 분석
    paper_title = list(final_summaries.keys())[0]
    paper_summary = final_summaries[paper_title]
    paper_domain = paper_domains[paper_title]

    # 논문 제목 단축 (로깅용)
    short_title = paper_title[:50] + "..." if len(paper_title) > 50 else paper_title
    logger.info(f"🔄 교차 도메인 분석 시작: '{short_title}' (ReAct with tools)")

    # 도메인 정보 로깅
    main_fields = paper_domain.get("main_field", [])
    sub_fields = paper_domain.get("sub_field", [])
    logger.info(f"📊 분석 대상 도메인 - 주요: {main_fields}, 세부: {sub_fields}")

    # --- Tools 구성 ---
    tools = []

    # 1) Web Search Tool (공통 함수 사용)
    web_tool = get_web_search_tool(max_results=5)
    if web_tool is not None:
        tools.append(web_tool)
    else:
        logger.warning("⚠️ Web Search Tool 비활성화 (사용 가능한 검색 툴 없음)")

    # 2) Vectorstore Search Tool (공통 함수 사용)
    vs = vectorstores.get(paper_title)
    vs_tool = get_vectorstore_search_tool(vs, paper_title)
    if vs_tool is not None:
        tools.append(vs_tool)
    else:
        logger.warning(f"⚠️ Vectorstore Search Tool 비활성화 (벡터스토어 없음: {paper_title})")

    # --- ReAct Agent 생성 ---
    agent = create_react_agent(CROSS_DOMAIN_LLM, tools=tools, state_modifier=instructions)

    # 분석 프롬프트 생성 (사용자 메시지)
    prompt = f"""
    Please provide a comprehensive cross-domain analysis report for the following paper:

    # Domain Information
    - Main Fields: {main_fields}
    - Sub Fields: {sub_fields}

    # Paper Summary
    ----------------------------
    {paper_summary}
    ----------------------------

    Follow the instruction and use tools (web search and vectorstore_search) when necessary to improve factual accuracy,
    verify definitions, and extract missing technical context from the source document. If the summary is sufficient, you may proceed without tools.
    """

    try:
        logger.info("🔍 ReAct 에이전트 실행 (tool 사용 가능)...")
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if messages:
            analysis_report = messages[-1].content
        else:
            # 일부 구현에서 invoke가 문자열을 직접 반환할 수 있음
            analysis_report = str(result)

        logger.info("✅ 교차 도메인 분석 완료")
        logger.debug(f"분석 결과 길이: {len(analysis_report)} 문자")

        return {
            **state,
            "analysis_report": analysis_report
        }

    except Exception as e:
        error_msg = f"교차 도메인 분석 중 오류 발생: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return {
            **state,
            "analysis_report": f"교차 도메인 분석 오류: {error_msg}"
        }