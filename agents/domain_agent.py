from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from src.state import State
from src.tracking import track_agent
import logging

# 현재 모듈 로거 생성
logger = logging.getLogger(__name__)

# instruction = """
# You are a Domain Identify Agent for academic papers. 
# Your task is to identify the main and sub-fields of a given paper based on its summary.
# """

## 도메인 판별 LLM
DOMAIN_IDENTIFY_LLM = ChatOpenAI(
    model = "gpt-4.1-mini",
    temperature = 0
)

class Domain(TypedDict):
    main_field: Annotated[list[str], [], "The main field of the paper. Ex. Computer Science, Physics, Psychology"]
    sub_field: Annotated[list[str], [], "The sub-field of the paper. Ex. Machine Learning, Quantum Physics, Cognitive Psychology"]

@track_agent("domain_agent")
def domain_agent(state: State) -> State:
    """도메인 판별 에이전트"""
    summaries = state["final_summary"]
    paper_titles = list(summaries.keys())
    
    logger.info(f"🔍 도메인 분석 시작... 분석 대상: {len(paper_titles)}개 논문")
    logger.info(f"📄 분석 논문: {paper_titles}")
    
    paper_domain = {}
    
    for paper_title, summary in summaries.items():
        # 논문 제목 단축 (로깅용)
        short_title = paper_title[:30] + "..." if len(paper_title) > 30 else paper_title
        logger.info(f"📑 도메인 분석 중...")
        
        try:
            structured_llm = DOMAIN_IDENTIFY_LLM.with_structured_output(Domain)
            domain = structured_llm.invoke(f"Set the main & sub fields of the paper based on summary:\n{state['final_summary'][paper_title]}.")
            paper_domain[paper_title] = domain
            
            logger.info(f"   ✅ {short_title} 분야: {paper_domain[paper_title]}")
            
        except Exception as e:
            error_msg = f"도메인 분석 중 오류 발생: {str(e)}"
            logger.error(f"   ❌ '{short_title}' {error_msg}")
            
            # 오류 발생시 기본값 설정
            paper_domain[paper_title] = {
                "main_field": ["Unknown"],
                "sub_field": ["Error in analysis"]
            }
    
    logger.info(f"✅ 도메인 분석 완료!")
    
    return {"paper_domain": paper_domain}