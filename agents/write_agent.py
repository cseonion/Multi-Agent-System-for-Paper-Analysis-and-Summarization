from src.state import State
from src.tracking import track_agent
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from agents.tools.python_repl import get_python_repl_tool
from config.agent_llm import get_llm

logger = logging.getLogger(__name__)

instructions = """
You are a technical blog writer (like Medium) who turns analysis into an engaging article for a broad technical audience.

# Output Format
    - Markdown only.
    - Always use proper Markdown structure: #, ##, ### headings, bullet/numbered lists, links, and fenced code blocks (```python ... ``` for code).
    - No HTML, no images. Keep content self-contained.

# Style
    - Tone: clear, informative, slightly narrative; avoid academic-report tone and marketing hype.
    - Audience: engineers and researchers across domains (assume mixed background knowledge).
    - Clarity: short paragraphs, descriptive headings, and smooth transitions between sections.
    - Terminology: define domain-specific terms briefly on first use.
    - Evidence & Assumptions: avoid hallucinating exact dataset names or results; if unsure, clearly state assumptions.
    - Neutrality: present limitations and trade-offs fairly; avoid exaggerated claims.
    - Language: concise sentences, active voice.

# Requirements
    1) Title
    - Start with a compelling, descriptive title (use a top-level # heading).
    2) Introduction
    - Provide motivation, context, and what readers will learn in a short intro.
    3) Body
    - Organize with clear headings (#, ##, ###) and concise explanations.
    - Use bullet lists for key points when helpful.
    4) Code Snippets (conditional)
    - Only include code blocks if and only if the paper domain is Computer Science (CS).
    - Prefer pseudo-code style sketches inspired by related work; keep examples minimal and self-contained.
    - Avoid heavy dependencies, external network calls, or unsafe operations.
    5) Conclusion & Takeaways
    - Summarize key insights and add bullet-point takeaways.
    - Mention potential applications or next steps.
    6) Safety & Accuracy
    - Do not fabricate dataset names, metrics, or results.
    - If numbers are unknown, use qualitative statements or clearly noted estimates.
    - Explicitly mark any assumptions.

# Constraints
    - Markdown only output. No HTML/images.
    - Keep lines reasonably short and the article readable on typical blog platforms.
"""

WRITER_LLM = get_llm("writer_agent")


@track_agent("write_agent")
def write_agent(state: State) -> State:
    """블로그 스타일 라이팅 에이전트.
    - analysis_plan == 'single'이면 요약본 기반 작성(리포트 없음 가능)
    - 그 외(cross_domain, comparison, literature_review)는 요약본+분석 리포트 동시 제공
    - 코드 스니펫은 CS 도메인에서만 허용(의사코드 중심)
    """
    analysis_plan = state.get("analysis_plan")
    summaries = state.get("final_summary", {}) or {}
    analysis_report = state.get("analysis_report")
    paper_domains = state.get("paper_domain", {}) or {}
    cache_dir = state.get("cache_dir")

    # 도메인 판단 (하나라도 CS면 True)
    def _is_cs_domain(dom: dict) -> bool:
        try:
            mains = " ".join(dom.get("main_field", [])).lower()
            subs = " ".join(dom.get("sub_field", [])).lower()
            return ("computer" in mains) or ("computer" in subs)
        except Exception:
            return False

    is_cs = any(_is_cs_domain(dom) for dom in paper_domains.values()) if paper_domains else False

    # 소스 구성: 요약본(단일/다중) + 리포트(조건부)
    if analysis_plan == "single":
        if not summaries:
            return {**state, "final_report": "요약본이 없어 글을 생성할 수 없습니다."}
        paper_title = next(iter(summaries.keys()))
        # 단일 논문 요약만 제공
        summaries_block = f"# Summary\n\n- Paper: {paper_title}\n\n{summaries[paper_title]}\n"
        report_block = ""  # single 플랜에서는 리포트가 없을 수 있음
        context_heading = f"Based on the paper summary: {paper_title}"
        base_name = paper_title
    else:
        # 다중 논문: 모든 요약본을 합쳐서 제공 + 분석 리포트 포함
        if not summaries:
            summaries_block = "# Summaries\n\n(요약본이 비어있습니다)\n"
        else:
            parts = []
            for t, s in summaries.items():
                parts.append(f"## {t}\n\n{s}")
            summaries_block = "\n".join(parts)
        report_block = f"\n{analysis_report}\n" if analysis_report else ""
        context_heading = "Based on the paper summaries and the analysis report"
        base_name = f"{analysis_plan or 'analysis'}"

    # 도구 구성 (Python REPL): CS 도메인에서만 활성화
    tools = []
    if is_cs:
        py_tool = get_python_repl_tool()
        if py_tool is not None:
            tools.append(py_tool)
            logger.info("🔧 Python REPL Tool 활성화 (CS 도메인)")
    else:
        logger.info("🚫 비-CS 도메인: 코드 스니펫/REPL 비활성화")

    agent = create_react_agent(WRITER_LLM, tools=tools, state_modifier=instructions)

    # 도메인 목록 문자열
    if paper_domains:
        domain_lines = []
        for t, dom in paper_domains.items():
            mains = ", ".join(dom.get("main_field", []))
            subs = ", ".join(dom.get("sub_field", []))
            domain_lines.append(f"- {t}: main=[{mains}] | sub=[{subs}]")
        domains_block = "\n".join(domain_lines)
    else:
        domains_block = "(도메인 정보 없음)"

    prompt = f"""
    - Context Heading: {context_heading}
    - Domains: {domains_block}
    
    # Summaries:  
    {summaries_block}
    
    
    # Analysis Report: 
    {report_block}
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
        return {**state, "final_report": f"글 생성 오류: {e}"}