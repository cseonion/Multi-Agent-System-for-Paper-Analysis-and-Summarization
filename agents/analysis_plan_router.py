from src.state import State
from src.tracking import track_agent
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Annotated
import logging
from langchain_core.prompts import ChatPromptTemplate

# 현재 모듈 로거 생성
logger = logging.getLogger(__name__)

instructions = """
You are an Analysis Plan Router Agent for academic papers.
You need to generate answers to the given question based on the domains of a paper or papers.
"""

ANALYSIS_PLAN_LLM = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)

class CrossDomain(TypedDict):
    is_cross_domain_paper: Annotated[bool, False, "Is the paper cross-domain?"]
    
class StronglyRelated(TypedDict):
    is_strongly_related: Annotated[bool, False, "If the domains of the input papers are strongly related to each other, i.e., if the papers are deeply related to each other, return True. If the input papers share a common topic but are not strongly related, return False"]

@track_agent("analysis_planner")
def analysis_planner(state: State):
    papers = state["paper_title"]
    domains = state["paper_domain"]
    
    logger.info(f"🎯 분석 계획 생성 중... 논문 수: {len(papers)}")
    
    if len(papers) == 1:
        paper_title = papers[0]
        logger.info(f"📖 단일 논문 분석: '{paper_title}'")
        
        def is_cross_domain(domains):
            # state["paper_domain"][paper_title]["main_field"] 기준으로 수정
            main_field = domains.get(paper_title, {}).get("main_field", [])
            prompt = ChatPromptTemplate.from_messages([
                ("system", instructions),
                ("human", "Main field of the paper:\n{main_field}")
            ])
            runner = prompt | ANALYSIS_PLAN_LLM.with_structured_output(CrossDomain)
            logger.debug(f"🔍 도메인 분석 중: {main_field}")
            return runner.invoke({"main_field": main_field})
        
        if is_cross_domain(domains)["is_cross_domain_paper"]:
            state["analysis_plan"] = "cross_domain"
            logger.info("✅ 분석 계획 결정: 교차 도메인 (cross_domain)")
            return "cross_domain"
        else:
            state["analysis_plan"] = "single"
            logger.info("✅ 분석 계획 결정: 단일 도메인 (single)")
            return "single"
            
    elif len(papers) > 1:
        logger.info(f"📚 다중 논문 분석: {len(papers)}개 논문")
        
        def papers_strongly_related(papers, domains):
            fields = {}
            for i, paper in enumerate(papers):
                field = domains.get(paper, {})
                fields[f"paper_{i}"] = field
            prompt = ChatPromptTemplate.from_messages([
                ("system", instructions),
                ("human", "Fields of the papers:\n{fields}")
            ])
            runner = prompt | ANALYSIS_PLAN_LLM.with_structured_output(StronglyRelated)
            logger.debug("🔍 논문 간 연관성 분석 중...")
            return runner.invoke({"fields": fields})
        
        if papers_strongly_related(papers, domains)["is_strongly_related"]:
            state["analysis_plan"] = "comparison"
            logger.info("✅ 분석 계획 결정: 비교 분석 (comparison)")
            return "comparison"
        else:
            state["analysis_plan"] = "literature_review"
            logger.info("✅ 분석 계획 결정: 문헌 리뷰 (literature_review)")
            return "literature_review"