import os
import re
import logging
from typing import List, cast
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from src.tracking import track_agent
from config.agent_llm import get_llm

from src.state import SummaryState

# LLM 모델 정의
## 섹션 요약 LLM
SECTION_SUMMARY_LLM = get_llm("summary_agent_section")
instruct_section = """
You are an expert research analyst summarizing academic papers section by section.
The summary will go sequentially from the first section to the last, and the goal is to summarise the key points of each section. To get started, you can use the following link:
You will be given two inputs:
    1. the raw text of the paper section
    2. a summary of the previous sections (but not when summarising the first section)

The previous section summaries are provided simply as a reference for continuity in your summarisation work, and **do not directly reflect the content of your summary**
In particular, be careful not to distort numerical information.
"""
## 최종 요약 LLM
FINAL_SUMMARY_LLM = get_llm("summary_agent_final")
instruct_final = """
You are an expert research analyst creating a comprehensive final summary of an academic paper.
Based on the section-by-section summaries written by other agents, you will write a final summary that includes.
Write a well-structured and comprehensive final summary that includes (IMPORTANT):
    1. **Research Objective and Background**
    2. **Key Methodology**
    3. **Result and Conclusion**
    4. **Implications and Significance**
    5. **Limitations, if any**

Please provide a well-structured, comprehensive final summary that synthesizes all sections into a coherent overview of the entire paper.
"""
# 현재 모듈 로거 생성 (main.py에서 설정한 로깅 사용)
logger = logging.getLogger(__name__)

def print_completion_message(paper_title: str, stage: str, details: str = ""):
    """
    단순한 완료 메시지 출력 함수
    
    Args:
        paper_title: 논문 제목
        stage: 처리 단계 (예: "섹션 추출", "섹션 요약", "최종 요약")
        details: 추가 정보 (예: 섹션 개수)
    """
    short_title = paper_title[:50] + "..." if len(paper_title) > 50 else paper_title
    message = f"✅ '{short_title}' {stage} 완료"
    if details:
        message += f" ({details})"
    print(message)

class SummaryFileManager:
    """요약 파일 저장 및 관리를 담당하는 클래스"""
    
    def __init__(self):
        pass
    
    def _sanitize_filename(self, filename: str) -> str:
        """파일명에서 특수문자 제거 및 정리"""
        # 특수문자를 언더스코어로 변경
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 연속된 공백을 하나로 변경
        filename = re.sub(r'\s+', ' ', filename)
        # 앞뒤 공백 제거
        filename = filename.strip()
        return filename
    
    def _get_summaries_dir_from_vectorstore_path(self, vectorstore_path: str) -> str:
        """vectorstore_path에서 summaries 디렉토리 경로 생성"""
        # vectorstore_path에서 한 폴더 위로 올라가기
        # 예: /path/to/datetime/paper_title/vectorstore → /path/to/datetime/paper_title/summaries
        paper_dir = os.path.dirname(vectorstore_path)
        summaries_dir = os.path.join(paper_dir+"/../", "summaries")
        
        # 디렉토리 생성
        os.makedirs(summaries_dir, exist_ok=True)
        
        return summaries_dir
    
    def save_section_summary(self, paper_title: str, section_number: str, 
                           section_name: str, summary_content: str, vectorstore_path: str) -> str:
        """개별 섹션 요약을 텍스트 파일로 저장"""
        
        # vectorstore_path에서 summaries 디렉토리 생성
        summaries_dir = self._get_summaries_dir_from_vectorstore_path(vectorstore_path)
        
        # 파일명 생성: "section_number section_name.txt"
        filename = f"{section_number} {section_name}.txt"
        safe_filename = self._sanitize_filename(filename)
        
        # 전체 파일 경로
        file_path = os.path.join(summaries_dir, safe_filename)
        
        try:
            # 요약 내용을 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                # f.write(f"논문 제목: {paper_title}\n")
                # f.write(f"섹션: {section_number}. {section_name}\n")
                # f.write("=" * 50 + "\n\n")
                f.write(summary_content)
                # f.write("\n\n")
                # f.write("=" * 50 + "\n")
                # f.write(f"생성 시간: {self._get_current_timestamp()}\n")
            
            # 개별 섹션 저장 메시지 제거 (섹션별 요약 완료시에만 표시)
            return file_path
            
        except Exception as e:
            print(f"      ❌ 섹션 요약 저장 실패: {str(e)}")
            return ""
    
    def save_final_summary(self, paper_title: str, final_summary: str, vectorstore_path: str) -> str:
        """최종 요약을 텍스트 파일로 저장"""
        
        # vectorstore_path에서 summaries 디렉토리 생성
        summaries_dir = self._get_summaries_dir_from_vectorstore_path(vectorstore_path)
        
        # 파일명: "Final_Summary.txt"
        filename = "Final_Summary.txt"
        file_path = os.path.join(summaries_dir, filename)
        
        try:
            # 최종 요약을 파일에 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                # f.write(f"논문 제목: {paper_title}\n")
                # f.write("=" * 50 + "\n\n")
                f.write(final_summary)
                # f.write("\n\n")
                # f.write("=" * 50 + "\n")
                # f.write(f"생성 시간: {self._get_current_timestamp()}\n")
            
            print(f"      💾 최종 요약 저장됨: {filename}")
            return file_path
            
        except Exception as e:
            print(f"      ❌ 최종 요약 저장 실패: {str(e)}")
            return ""
    
    def save_all_summaries_index(self, paper_title: str, section_summaries: List[str], 
                               paper_sections: List[tuple], vectorstore_path: str) -> str:
        """모든 요약을 하나의 인덱스 파일로 저장"""
        
        # vectorstore_path에서 summaries 디렉토리 생성
        summaries_dir = self._get_summaries_dir_from_vectorstore_path(vectorstore_path)
        
        # 파일명: "00 Index_All_Summaries.txt"
        filename = "00 Index_All_Summaries.txt"
        file_path = os.path.join(summaries_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"논문 제목: {paper_title}\n")
                f.write("전체 섹션별 요약 인덱스\n")
                f.write("=" * 60 + "\n\n")
                
                # 각 섹션별 요약 작성
                for i, summary in enumerate(section_summaries):
                    if i < len(paper_sections):
                        section_number, section_name = paper_sections[i]
                        f.write(f"섹션 {section_number}: {section_name}\n")
                    else:
                        f.write(f"섹션 {i+1}: 추가 섹션\n")
                    
                    f.write("-" * 40 + "\n")
                    f.write(summary)
                    f.write("\n\n")
                
                f.write("=" * 60 + "\n")
                f.write(f"총 섹션 수: {len(section_summaries)}\n")
                f.write(f"생성 시간: {self._get_current_timestamp()}\n")
            
            print(f"      📑 전체 요약 인덱스 저장됨: {filename}")
            return file_path
            
        except Exception as e:
            print(f"      ❌ 인덱스 파일 저장 실패: {str(e)}")
            return ""
    
    def _get_current_timestamp(self) -> str:
        """현재 시간 문자열 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class SequentialSummaryAgent:
    """연속적 섹션 요약을 수행하는 Summary Agent"""
    
    def __init__(self):
        # 각 작업에 맞는 LLM 모델 설정
        self.section_llm = SECTION_SUMMARY_LLM
        self.final_llm = FINAL_SUMMARY_LLM
    
    def create_section_summary_prompt(self, current_section: str, section_name: str, 
                                     previous_summary: str = "", paper_title: str = "") -> str:
        """연속적 섹션 요약을 위한 프롬프트 생성"""
        
        base_prompt = f"""
        # Current Section Title: {section_name}
        # Current Section Content:
        {current_section}
        """

        if previous_summary:
            continuity_prompt = f"""
            
            # Previous Section Summary for Context:
            {previous_summary}
            """
            base_prompt += continuity_prompt

        # base_prompt += "\n\nProvide your summary in Korean:"
        return base_prompt

    def create_final_summary_prompt(self, section_summaries: List[str], paper_title: str) -> str:
        """최종 요약을 위한 프롬프트 생성"""
        combined_sections = "\n\n".join([f"섹션 {i+1}: {summary}" for i, summary in enumerate(section_summaries)])
        
        return f"""
        # Paper Title: {paper_title}
        # Combined Section Summaries:
        {combined_sections}
        """


class SummaryProcessor:
    """Summary 처리를 담당하는 메인 프로세서"""
    
    def __init__(self):
        # 각 작업에 맞는 LLM과 agent 초기화
        self.section_llm = SECTION_SUMMARY_LLM
        self.final_llm = FINAL_SUMMARY_LLM
        self.agent = SequentialSummaryAgent()
        self.file_manager = SummaryFileManager()
    
    @track_agent("extract_sections")
    def extract_sections_optimized(self, state: SummaryState) -> SummaryState:
        """
        이미 로드된 벡터스토어를 사용하여 섹션별 내용 추출
        벡터스토어 재로딩 없이 기존 검증된 로직 사용
        """
        paper_title = state["paper_title"]
        vectorstore = state.get("vectorstore")  # 이미 로드된 벡터스토어 사용
        paper_sections = state.get("paper_sections", [])
        
        # 이미 로드된 벡터스토어가 없으면 에러
        if vectorstore is None:
            raise ValueError(f"벡터스토어가 로드되지 않았습니다: {paper_title}")
        
        # 섹션별 내용 추출 (기존 검증된 로직 사용)
        extracted_sections = {}
        
        if not paper_sections:
            # 섹션 정보가 없는 경우 전체 문서 검색
            all_docs = vectorstore.similarity_search("", k=100)
            all_content = "\n\n".join([doc.page_content for doc in all_docs])
            extracted_sections["전체 문서"] = all_content
        else:
            # 섹션별로 문서 추출 (기존 로직 그대로)
            for i, (section_number, section_name) in enumerate(paper_sections, 1):
                try:
                    # 해당 섹션에 속하는 문서들 검색
                    section_docs = []
                    
                    # vectorstore의 모든 문서를 가져와서 metadata로 필터링
                    all_docs = vectorstore.similarity_search("", k=1000)
                    
                    for doc in all_docs:
                        doc_metadata = doc.metadata
                        
                        # metadata에서 section_number와 section 정보 확인
                        doc_section_number = doc_metadata.get("section_number", "")
                        doc_section = doc_metadata.get("section", "")
                        
                        # 섹션 번호 또는 섹션 이름이 일치하는 문서 수집
                        if (str(doc_section_number) == str(section_number) or 
                            doc_section.strip().lower() == section_name.strip().lower()):
                            section_docs.append(doc)
                    
                    if section_docs:
                        # 섹션 문서들의 내용을 결합
                        section_content = "\n\n".join([doc.page_content for doc in section_docs])
                        extracted_sections[f"{section_number}. {section_name}"] = section_content
                    else:
                        # 섹션 이름으로 유사도 검색 시도
                        similar_docs = vectorstore.similarity_search(section_name, k=5)
                        if similar_docs:
                            section_content = "\n\n".join([doc.page_content for doc in similar_docs])
                            extracted_sections[f"{section_number}. {section_name}"] = section_content
                        else:
                            extracted_sections[f"{section_number}. {section_name}"] = f"섹션 '{section_name}'에 해당하는 내용을 찾을 수 없습니다."
                            
                except Exception as e:
                    extracted_sections[f"{section_number}. {section_name}"] = f"섹션 추출 중 오류 발생: {str(e)}"
        
        print_completion_message(paper_title, "섹션 추출", f"{len(extracted_sections)}개 섹션")
        
        return cast(SummaryState, {
            **state,
            "sections": extracted_sections,
            "section_summaries": [],
            "final_summary": ""
        })
    
    @track_agent("summarize_sections")
    def summarize_sections_sequential(self, state: SummaryState) -> SummaryState:
        """섹션별 순차 요약 (이전 요약 컨텍스트 활용)"""
        # SummaryState 구조에 맞게 데이터 추출
        paper_title = state["paper_title"]  # SummaryState에서는 단일 문자열
        sections = state["sections"]
        paper_sections = state.get("paper_sections", [])  # SummaryState에서는 리스트
        
        # 섹션 요약 시작 메시지
        short_title = paper_title[:50] + "..." if len(paper_title) > 50 else paper_title
        print(f"⏳ '{short_title}' 섹션 요약 중... ({len(sections)}개 섹션)")
        
        section_summaries = []
        previous_summary = ""
        
        for i, (section_name, section_content) in enumerate(sections.items(), 1):
            # 섹션 요약 생성
            prompt = self.agent.create_section_summary_prompt(
                current_section=section_content,
                section_name=section_name,
                previous_summary=previous_summary,
                paper_title=paper_title
            )
            
            try:
                response = self.section_llm.invoke([
                    SystemMessage(content=instruct_section),
                    HumanMessage(content=prompt)
                ])
                
                current_summary = response.content if hasattr(response, 'content') else str(response)
                section_summaries.append(current_summary)
                
                # 섹션 요약을 파일로 저장
                if self.file_manager:
                    # 섹션 번호와 이름 추출 (인덱스 수정: i-1 사용)
                    if (i-1) < len(paper_sections):
                        section_number, section_name_clean = paper_sections[i-1]  # i는 1부터 시작하므로 i-1
                    else:
                        # paper_sections에서 정보를 가져올 수 없는 경우 section_name에서 추출
                        parts = section_name.split('. ', 1)
                        if len(parts) == 2:
                            section_number, section_name_clean = parts
                        else:
                            section_number = str(i + 1)
                            section_name_clean = section_name
                    
                    # 파일로 저장 (vectorstore_path 전달)
                    self.file_manager.save_section_summary(
                        paper_title=paper_title,
                        section_number=section_number,
                        section_name=section_name_clean,
                        summary_content=current_summary,
                        vectorstore_path=state["vectorstore_path"]
                    )
                
                # 다음 섹션을 위해 이전 요약 업데이트 (최근 2개 섹션 요약만 유지)
                if len(section_summaries) >= 2:
                    previous_summary = section_summaries[-1]  # 가장 최근 요약만 사용
                elif len(section_summaries) == 1:
                    previous_summary = section_summaries[0]
                    
            except Exception as e:
                error_msg = f"섹션 '{section_name}' 요약 중 오류: {str(e)}"
                section_summaries.append(error_msg)
        
        # 모든 섹션 요약을 인덱스 파일로 저장
        if self.file_manager and section_summaries:
            self.file_manager.save_all_summaries_index(
                paper_title=paper_title,
                section_summaries=section_summaries,
                paper_sections=paper_sections,
                vectorstore_path=state["vectorstore_path"]
            )
        
        print_completion_message(paper_title, "섹션별 요약", f"{len(section_summaries)}개 섹션")
        
        return cast(SummaryState, {
            **state,
            "section_summaries": section_summaries
        })
    
    @track_agent("create_final_summary")
    def create_final_summary(self, state: SummaryState) -> SummaryState:
        """섹션 요약들을 기반으로 최종 요약 생성"""
        # SummaryState 구조에 맞게 데이터 추출
        paper_title = state["paper_title"]  # SummaryState에서는 단일 문자열
        section_summaries = state["section_summaries"]
        
        if not section_summaries:
            return cast(SummaryState, {
                **state,
                "final_summary": f"'{paper_title}' 섹션 요약이 없어 최종 요약을 생성할 수 없습니다."
            })
        
        prompt = self.agent.create_final_summary_prompt(section_summaries, paper_title)
        
        try:
            response = self.final_llm.invoke([
                SystemMessage(content=instruct_final),
                HumanMessage(content=prompt)
            ])
            
            final_summary = response.content if hasattr(response, 'content') else str(response)
            
            # 최종 요약을 파일로 저장
            if self.file_manager:
                self.file_manager.save_final_summary(
                    paper_title=paper_title,
                    final_summary=final_summary,
                    vectorstore_path=state["vectorstore_path"]
                )
            
            print_completion_message(paper_title, "최종 요약")
            
            return cast(SummaryState, {
                **state,
                "final_summary": final_summary
            })
            
        except Exception as e:
            error_msg = f"최종 요약 생성 중 오류: {str(e)}"
            print(f"❌ '{paper_title}' 최종 요약 오류: {error_msg}")
            
            return cast(SummaryState, {
                **state,
                "final_summary": error_msg
            })