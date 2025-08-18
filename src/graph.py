from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Dict, Any, Sequence

from src.load_doc import extract
from agents.summary_agent import SummaryProcessor
from agents.domain_agent import domain_agent
from agents.analysis_plan_router import analysis_planner
from agents.analysis_cross_domain_agent import cross_domain_agent
from agents.analysis_comparison_agent import comparison_agent
from agents.analysis_lit_review_agent import lit_review_agent
from agents.write_agent import write_agent
from src.state import State, SummaryState

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
import logging
import traceback

# 현재 모듈 로거 생성 (main.py에서 설정한 로깅 사용)  
logger = logging.getLogger(__name__)

class GraphWorkflow:
    def __init__(self):
        self.processor = SummaryProcessor()
        self.compiled_summary_subgraph = None  # subgraph 참조를 저장

    def summary_subgraph(self):
        """개별 논문 처리를 위한 Summary Subgraph"""
        
        # Subgraph 상태 정의 - SummaryState 형태로 처리
        subgraph_flow = StateGraph(SummaryState)
        
        # 노드 추가 - 최적화된 섹션 추출 사용 (벡터스토어 재로딩 없음)
        subgraph_flow.add_node("extract_sections", self.processor.extract_sections_optimized)
        subgraph_flow.add_node("summarize_sections", self.processor.summarize_sections_sequential) 
        subgraph_flow.add_node("create_final_summary", self.processor.create_final_summary)

        # 부모 State로 결과를 전달할 수 있도록 포장하는 노드 추가
        def _pack_parent_update(state: SummaryState) -> Dict[str, Dict[str, str]]:
            """Subgraph 결과를 부모 State로 전달할 수 있게 key를 맞춰 포장.
            부모 State에는 Dict[str, str] 형태의 요약본 모음(예: summaries)이 있어야 하며,
            해당 key의 reducer는 dict 병합(예: operator.or_)이어야 병렬 병합이 안전합니다.
            """
            paper_title = state.get("paper_title", "")
            section_summaries = state.get("section_summaries", {})
            final_summary = state.get("final_summary", "")
            sections = state.get("sections", {})
            
            # 부모 State의 병합 대상 key 이름은 'summaries'로 가정합니다.
            return {
                "section_summaries": {paper_title: section_summaries},
                "final_summary": {paper_title: final_summary},
                "section_texts": {paper_title: sections},
                }
        subgraph_flow.add_node("pack_parent_update", _pack_parent_update)
        
        # 엣지 구성
        subgraph_flow.add_edge(START, "extract_sections")
        subgraph_flow.add_edge("extract_sections", "summarize_sections")
        subgraph_flow.add_edge("summarize_sections", "create_final_summary")
        subgraph_flow.add_edge("create_final_summary", "pack_parent_update")
        subgraph_flow.add_edge("pack_parent_update", END)
        
        # 컴파일된 subgraph를 인스턴스 변수에 저장
        compiled_subgraph = subgraph_flow.compile()
        self.compiled_summary_subgraph = compiled_subgraph
        return compiled_subgraph
    

    def create_parallel_summary_nodes(self):
        """병렬 Summary 처리를 위한 노드들 (Send API + 올바른 State 변환)"""
        
        def initiate_parallel_summaries(state: State) -> Sequence[Send]:
            """병렬 처리를 위해 각 논문을 개별 subgraph로 전송"""
            
            paper_titles = state["paper_title"]
            vectorstores = state["vectorstores"]
            vectorstores_path = state["vectorstores_path"]
            paper_sections = state.get("paper_sections", {})
            cache_dir = state.get("cache_dir", "")
            
            logger.info(f"🚀 {len(paper_titles)}개 논문에 대한 병렬 요약 작업 시작...")
            
            # 각 논문에 대해 Send 객체 생성 - SummaryState 형태로 전송
            send_list = []
            for paper_title in paper_titles:
                paper_state = {
                    "paper_title": paper_title,
                    "paper_sections": paper_sections.get(paper_title, []), 
                    "vectorstore": vectorstores.get(paper_title),
                    "vectorstore_path": vectorstores_path.get(paper_title, ""),
                    "cache_dir": cache_dir,
                    "sections": {},  # 초기값
                    "section_summaries": [],  # 초기값
                    "final_summary": ""  # 초기값
                }
                
                # Summary subgraph로 전송
                send_list.append(Send("summary_subgraph", paper_state))
                logger.info(f"  📤 '{paper_title}' → Summary Subgraph로 전송 (섹션 {len(paper_sections.get(paper_title, []))}개)")
            
            return send_list

        return initiate_parallel_summaries

    def build_workflow(self):
        """Send API 기반 병렬 처리 워크플로우 (개선된 버전)"""
        
        logger.info("🔧 Send API 기반 병렬 워크플로우 구성 시작...")
        
        # Summary subgraph 생성
        summary_subgraph = self.summary_subgraph()

        # Subgraph 래퍼: 부모로는 필요한 키만 전달하여 key 충돌(예: paper_title) 방지
        def run_summary_subgraph(sub_state: SummaryState) -> Dict[str, Any]:
            result = summary_subgraph.invoke(sub_state)
            out: Dict[str, Any] = {}
            if "section_summaries" in result:
                out["section_summaries"] = result["section_summaries"]
            if "final_summary" in result:
                out["final_summary"] = result["final_summary"]
            if "section_texts" in result:
                out["section_texts"] = result["section_texts"]
            return out
        
        # 병렬 처리 노드 함수 생성
        initiate_parallel_summaries = self.create_parallel_summary_nodes()

        # 분석 플랜을 상태에 기록하는 노드 (상태 업데이트 전용)
        def compute_analysis_plan(state: State) -> Dict[str, Any]:
            # analysis_planner는 분기 문자열을 반환
            plan = analysis_planner(state)
            logger.info(f"🧭 분석 플랜 확정: {plan}")
            return {"analysis_plan": plan}

        # 메인 워크플로우 생성
        workflow = StateGraph(State)
        
        # 노드 추가
        workflow.add_node("extract", extract)
        workflow.add_node("summary_subgraph", run_summary_subgraph)  # 병렬 subgraph (래퍼)
        workflow.add_node("domain_agent", domain_agent)
        workflow.add_node("analysis_planner", compute_analysis_plan)  # 상태 업데이트 노드
        workflow.add_node("cross_domain_agent", cross_domain_agent)  # 교차 도메인 처리 노드
        workflow.add_node("comparison_agent", comparison_agent)
        workflow.add_node("lit_review_agent", lit_review_agent)  # 문헌 리뷰 처리 노드
        # workflow.add_node("ideation_agent", ideation_agent)  # 아이디어 생성 노드
        workflow.add_node("write_agent", write_agent)  # 단일 도메인 처리 노드
        
        # 엣지 구성
        workflow.add_edge(START, "extract")
        workflow.add_conditional_edges("extract", initiate_parallel_summaries, ["summary_subgraph"]) # Send API 병렬 처리
        workflow.add_edge("summary_subgraph", "domain_agent")  # LangGraph가 자동으로 결과 병합
        workflow.add_edge("domain_agent", "analysis_planner")  # 먼저 플랜을 상태에 기록
        # 상태 값 기반 분기
        workflow.add_conditional_edges(
            "analysis_planner",
            lambda s: s.get("analysis_plan"),
            {
                "single": "write_agent",
                "cross_domain": "cross_domain_agent",
                "comparison": "comparison_agent",
                "literature_review": "lit_review_agent",
            }
        )
        workflow.add_edge("cross_domain_agent", "write_agent")
        workflow.add_edge("comparison_agent", "write_agent")
        # workflow.add_edge("lit_review_agent", "ideation_agent")
        # workflow.add_edge("ideation_agent", "write_agent")
        workflow.add_edge("lit_review_agent", "write_agent")
        workflow.add_edge("write_agent", END)

        logger.info("✅ Send API 기반 병렬 워크플로우 구성 완료")
        return workflow.compile()

    # Workflow 시각화
    def visualize_workflow(self, app):
        try:
            display(Image(app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API,)))
        except Exception as e:
            logging.error(f"Visualization Error: {traceback.format_exc()}")
    
    def visualize_summary_subgraph(self):
        """Summary subgraph를 시각화"""
        if self.compiled_summary_subgraph:
            try:
                display(Image(self.compiled_summary_subgraph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API,)))
            except Exception as e:
                logging.error(f"Summary Subgraph Visualization Error: {traceback.format_exc()}")
        else:
            print("Summary subgraph가 아직 생성되지 않았습니다. build_workflow()를 먼저 실행하세요.")
    
    def get_summary_subgraph(self):
        """컴파일된 summary subgraph 반환"""
        return self.compiled_summary_subgraph
