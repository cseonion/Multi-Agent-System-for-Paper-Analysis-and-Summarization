from typing import TypedDict, Any, Annotated
import operator

class State(TypedDict):
    """
    extracted results from the paper
    """
    path: list[str]
    paper_title: list[str]  # Papers information
    paper_sections: dict[str, list[tuple[str, str]]]  # Sections of the paper
    vectorstores:  dict[str, Any]   # Vector stores for each paper
    vectorstores_path: dict[str, str]  # Paths to vector stores for each paper
    cache_dir: str  # Path to the cache directory
    
    """
    Collected for all submitted papers - Annotated for parallel updates
    """
    section_summaries: Annotated[dict[str, list[str]], operator.or_]  # 병렬 업데이트 지원
    final_summary: Annotated[dict[str, str], operator.or_] # 병렬 업데이트 지원
    # 섹션 원문 텍스트 저장 (paper_title -> {"N. Name": section_text})
    section_texts: Annotated[dict[str, dict[str, str]], operator.or_]
    methodology: dict[str, str]  # Key methodology used in the paper
    experiment: dict[str, str]  # experimental details
    results_and_conclusions: dict[str, str] # Results and conclusions of the paper
    paper_domain: dict[str, dict[str, str]]  # Domain of the paper
    
    """
    only for multiple papers
    """
    analysis_plan: str
    analysis_report: str
    final_report: str

# subgraph state for summarization
class SummaryState(TypedDict):
    paper_title: str  # Papers 
    paper_sections: list[tuple[str, str]]  # [('1', 'Introduction'), ('2', 'Related Work'), ...]
    vectorstore: Any   # Vector stores for each paper
    vectorstore_path: str  # Paths to vector stores for each paper
    cache_dir: str  # Path to the cache directory
    sections: dict[str, str] # Section name: section content (extracted results)
    section_summaries: list[str]  # Sections of the summary
    final_summary: str # Summary of the paper