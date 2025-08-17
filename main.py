import logging
import os
from datetime import datetime
from src.graph import GraphWorkflow
from src.tracking import init_tracking, finalize_tracking
import re


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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{_safe_name(name_base)}_{suffix}_{ts}.md"
        path = os.path.join(cache_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logging.getLogger(__name__).info(f"💾 저장됨: {path}")
        return path
    except Exception as e:
        logging.getLogger(__name__).error(f"파일 저장 실패({_safe_name(name_base)}): {e}")
        return ""


def run(path, print_log: bool = True, output_language: str = "en"):
    """
    논문 처리 워크플로우를 실행하는 메인 함수
    
    Args:
        path: 처리할 PDF 경로 (리스트)
        print_log (bool): 콘솔에 로그를 출력할지 여부. True이면 콘솔과 파일 모두에 출력, False이면 파일에만 출력
    
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
        
        logger.info("✅ Task Paper Processing Completed Successfully")
        return result, app, workflow
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        logger.error(f"🔍 Error details: {repr(e)}")
        raise e