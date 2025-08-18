import logging
import os
from datetime import datetime
from src.graph import GraphWorkflow
from src.tracking import init_tracking, finalize_tracking
import re
from src.eval import evaluate_with_judge  
from src.eval import rouge_l, bertscore_f1_single 
import json
from typing import Any, List 


def setup_logging(cache_dir: str = None, print_log: bool = True):
    """
    전체 프로젝트의 로깅을 설정하는 함수
    
    Args:
        cache_dir (str): 로그 파일을 저장할 캐시 디렉토리 경로
        print_log (bool): 콘솔에 로그를 출력할지 여부
    
    Returns:
        str: 로그 파일 경로 (파일 로그가 활성화된 경우)
    """
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 모든 핸들러 완전 제거
    root_logger.handlers.clear()
    
    log_file_path = None
    
    # 파일 로깅 설정 (항상 추가)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        log_file_path = os.path.join(cache_dir, "process.log")
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    
    # 콘솔 로깅은 print_log=True일 때만 추가
    if print_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger.addHandler(console_handler)
        
    # 전파 방지 - 이게 중요!
    root_logger.propagate = False
    
    # 파일 로깅 설정 (cache_dir이 제공된 경우)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        log_file_path = os.path.join(cache_dir, "process.log")
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        
        if print_log:
            # 로그 파일 초기화 메시지는 한 번만 출력
            pass
    
    return log_file_path
# output_language: en과 ko 중에서만 선택 가능

def _safe_name(base: str) -> str:
    base = re.sub(r"[^\w\- ]+", "", str(base)).strip().replace(" ", "_")
    return base[:80] if base else "output"


def _save_markdown(cache_dir: str, name_base: str, content: str, suffix: str) -> str:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        filename = "report.md"
        path = os.path.join(cache_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.getLogger(__name__).info(f"💾 저장됨: {path}")
        return path
    except Exception as e:
        logging.getLogger(__name__).error(f"파일 저장 실패({_safe_name(name_base)}): {e}")
        return ""


def _save_json(cache_dir: str, name_base: str, data: dict, suffix: str) -> str:
    """평가 결과 저장(로깅 최소화)."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        filename = "eval.json"
        path = os.path.join(cache_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return ""


def run(path, print_log: bool = True, output_language: str = "en", eval: bool = False):
    """
    논문 처리 워크플로우를 실행하는 메인 함수
    
    Args:
        path: 처리할 PDF 경로 (리스트)
        print_log (bool): 콘솔에 로그를 출력할지 여부. True이면 콘솔과 파일 모두에 출력, False이면 파일에만 출력
        eval (bool): True면 평가 지표를 생성하고 cache 디렉토리에 저장
    
    Returns:
        dict: 처리 결과
    """
    # 캐시 디렉토리 생성 (로그 파일용)
    cache_dir = "cache/" + f"{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(cache_dir, exist_ok=True)
    # 로깅 설정
    log_file_path = setup_logging(cache_dir=cache_dir, print_log=print_log)
    
    # 메인 로거 생성
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Task Paper Processing Started. Cache directory: {cache_dir}")
    logger.info(f"📋 Process logs are being recorded")
    logger.info(f"📂 Input path: {path}")
    
    if log_file_path:
        logger.info(f"📋 Log file: {log_file_path}")
    
    if not print_log:
        logger.info("🔇 Console logging disabled - logs will only be saved to file")
    
    try:
        # 추적 시스템 초기화
        init_tracking(cache_dir)
        
        # 워크플로우 실행
        workflow = GraphWorkflow()
        app = workflow.build_workflow()
        
        logger.info("⚙️  Starting workflow execution...")
        # cache_dir을 state에 포함하여 전달
        initial_state = {
            "path": path,
            "cache_dir": cache_dir
        }
        result = app.invoke(initial_state)
        
        # 추적 정보 마무리 및 요약 생성
        tracking_summary = finalize_tracking()
        if tracking_summary:
            logger.info(f"📊 실행 추적 완료 - 에이전트 {tracking_summary['total_agents']}개, 총 시간 {tracking_summary['total_duration_seconds']:.2f}초")
            result["tracking_summary"] = tracking_summary
        
        # 결과 저장: analysis_report와 final_report를 md로 저장
        analysis_plan = result.get("analysis_plan")
        # 저장용 이름 베이스 결정
        if analysis_plan == "single":
            # 단일인 경우 summary의 첫 논문 제목 사용
            summaries = result.get("final_summary", {}) or {}
            name_base = next(iter(summaries.keys()), analysis_plan or "result")
        else:
            name_base = analysis_plan or "analysis"
        
        # analysis_report 우선 저장, 없으면 analysis_result로 폴백
        analysis_text = result.get("analysis_report") or result.get("analysis_result")
        if analysis_text:
            _save_markdown(cache_dir, name_base, analysis_text, "analysis_report")
        else:
            logger.info("analysis_report 없음 - 저장 건너뜀")
        
        if result.get("final_report"):
            _save_markdown(cache_dir, name_base, result["final_report"], "final_report")
        else:
            logger.info("final_report 없음 - 저장 건너뜀")
        
        # ---------- 평가 (옵션) ----------
        if eval:
            eval_outputs: dict[str, dict[str, Any]] = {}

            # 공통 evidence: 각 논문의 최종 요약
            final_summaries: dict = result.get("final_summary", {}) or {}
            final_summary_texts: List[str] = list(final_summaries.values()) if isinstance(final_summaries, dict) else []

            # 1) domain_agent (judge)
            if result.get("paper_domain"):
                candidate = json.dumps(result["paper_domain"], ensure_ascii=False)
                r = evaluate_with_judge("domain_agent", candidate=candidate, evidences=final_summary_texts)
                eval_outputs["domain_agent"] = {"target": r.target, "scores": r.scores, "details": r.details}

            # 2) analysis_plan_router (judge)
            if result.get("analysis_plan"):
                titles = list(final_summaries.keys())
                evidence_text = f"num_papers={len(titles)}; titles={', '.join(titles[:5])}"
                r = evaluate_with_judge("analysis_plan_router", candidate=result["analysis_plan"], evidences=[evidence_text])
                eval_outputs["analysis_plan_router"] = {"target": r.target, "scores": r.scores, "details": r.details}

            # 3) summary_agent(section): 섹션 원문과 1:1 비교, LLM 호출 없음
            section_summaries: dict = result.get("section_summaries", {}) or {}
            section_texts: dict = result.get("section_texts", {}) or {}
            per_section_details: List[dict] = []
            per_section_avg_scores: List[float] = []

            def _sec_sort_key(k: str) -> int:
                m = re.match(r"^\s*(\d+)\.", k)
                return int(m.group(1)) if m else 10**9

            for paper_title, summaries in section_summaries.items():
                originals_map = section_texts.get(paper_title, {}) or {}
                if not originals_map:
                    continue
                keys_sorted = sorted(list(originals_map.keys()), key=_sec_sort_key)
                originals_seq = [originals_map[k] for k in keys_sorted]
                # 길이 맞추기
                n = min(len(summaries), len(originals_seq))
                for i in range(n):
                    pred = summaries[i]
                    ref = originals_seq[i]
                    rl = rouge_l(pred, ref)
                    bf = bertscore_f1_single(pred, ref)
                    per_section_details.append({
                        "paper": paper_title,
                        "section": keys_sorted[i] if i < len(keys_sorted) else f"{i+1}",
                        "rouge_l": rl,
                        "bertscore_f1": bf,
                    })
                    per_section_avg_scores.append((rl + bf) / 2.0)

            eval_outputs["summary_agent(section)"] = {
                "target": "summary_agent(section)",
                "scores": {"avg_score": (sum(per_section_avg_scores) / len(per_section_avg_scores)) if per_section_avg_scores else 0.0},
                "details": {"per_section": per_section_details},
            }

            # 4) summary_agent(final): 최종 요약 vs 섹션 요약 합본, 1:1 비교 (논문별 산출 후 평균)
            per_paper_details: List[dict] = []
            rl_vals: List[float] = []
            bf_vals: List[float] = []
            for paper_title, final_sum in final_summaries.items():
                sec_sums = section_summaries.get(paper_title, [])
                ref = "\n\n".join(sec_sums)
                if not final_sum or not ref:
                    continue
                rl = rouge_l(final_sum, ref)
                bf = bertscore_f1_single(final_sum, ref)
                per_paper_details.append({"paper": paper_title, "rouge_l": rl, "bertscore_f1": bf})
                rl_vals.append(rl)
                bf_vals.append(bf)
            avg_rl = (sum(rl_vals) / len(rl_vals)) if rl_vals else 0.0
            avg_bf = (sum(bf_vals) / len(bf_vals)) if bf_vals else 0.0
            eval_outputs["summary_agent(final)"] = {
                "target": "summary_agent(final)",
                "scores": {"rouge_l": avg_rl, "bertscore_f1": avg_bf},
                "details": {"per_paper": per_paper_details},
            }

            # 5) 분석 리포트 (judge)
            analysis_agent_map = {
                "comparison": "comparison_agent",
                "cross_domain": "cross_domain_agent",
                "literature_review": "lit_review_agent",
            }
            analysis_plan = result.get("analysis_plan")
            if result.get("analysis_report") and analysis_plan in analysis_agent_map:
                agent_name = analysis_agent_map[analysis_plan]
                r = evaluate_with_judge(agent_name, candidate=result["analysis_report"], evidences=final_summary_texts)
                eval_outputs[agent_name] = {"target": r.target, "scores": r.scores, "details": r.details}

            # 6) 최종 글(write_agent) (judge + 구조/가독성)
            if result.get("final_report"):
                evs = [result.get("analysis_report")] if result.get("analysis_report") else final_summary_texts
                r = evaluate_with_judge("write_agent", candidate=result["final_report"], evidences=evs, metrics=["structure", "readability"])
                eval_outputs["write_agent"] = {"target": r.target, "scores": r.scores, "details": r.details}

            # 저장 (경로만 반환, 추가 로깅 없음)
            _save_json(cache_dir, name_base, eval_outputs, "eval_results")
            # 완료 로깅만
            logger.info("✅ 평가 완료")
        
        logger.info("✅ Task Paper Processing Completed Successfully")
        return result, app, workflow
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        logger.error(f"🔍 Error details: {repr(e)}")
        raise e