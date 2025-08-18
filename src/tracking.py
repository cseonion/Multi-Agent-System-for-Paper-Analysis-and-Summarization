import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from functools import wraps
import logging
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# 현재 실행 중인 에이전트/모델 및 실행 객체 (LLM 콜백에서 사용)
_current_agent_name: ContextVar[Optional[str]] = ContextVar("_current_agent_name", default=None)
_current_execution: ContextVar[Optional["AgentExecution"]] = ContextVar("_current_execution", default=None)
_current_model_name: ContextVar[Optional[str]] = ContextVar("_current_model_name", default=None)

try:
    # LangChain 콜백 베이스 (LC 0.3+)
    from langchain_core.callbacks import BaseCallbackHandler  # type: ignore
except Exception:  # 호환성 대비
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
    except Exception:
        class BaseCallbackHandler:  # 최소 폴백
            pass

# 설정 로더 (단가 맵 참조용)
try:
    from config.agent_llm import load_agent_config
except Exception:
    def load_agent_config():
        return {}


@dataclass
class AgentExecution:
    """에이전트 실행 정보를 담는 데이터클래스"""
    agent_name: str
    start_time: str
    end_time: str
    duration_seconds: float
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    token_cost_usd: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class WorkflowTracking:
    """전체 워크플로우 추적 정보"""
    workflow_id: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_seconds: Optional[float] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    agent_executions: List[AgentExecution] = None
    
    def __post_init__(self):
        if self.agent_executions is None:
            self.agent_executions = []


class ExecutionTracker:
    """에이전트 실행을 추적하는 클래스"""
    
    # ---- 단가 맵 유틸 ----
    @staticmethod
    def _get_model_pricing_per_million(model_name: Optional[str]) -> tuple[float, float]:
        """config.agent_config.json의 pricing_per_million에서 (input, output)을 USD로 반환
        - 사용자가 입력한 값을 그대로 사용
        - 항목이 없거나 형식이 잘못되면 0으로 처리 (로그 출력 없음)
        """
        config = load_agent_config() or {}
        pricing = (config.get("pricing_per_million") or {}) if isinstance(config, dict) else {}
        if not model_name:
            return 0.0, 0.0
        model_pr = pricing.get(model_name)
        if not isinstance(model_pr, dict):
            return 0.0, 0.0
        inp = model_pr.get("input")
        outp = model_pr.get("output")
        if isinstance(inp, (int, float)) and isinstance(outp, (int, float)):
            return float(inp), float(outp)
        return 0.0, 0.0

    # ---- 비용 계산 ----
    @staticmethod
    def _calculate_token_cost(input_tokens: int, output_tokens: int, model_name: Optional[str] = None) -> float:
        """토큰 비용 계산 (USD/1M 기준 단가를 그대로 사용)"""
        input_price_per_million, output_price_per_million = ExecutionTracker._get_model_pricing_per_million(model_name)
        input_cost = (input_tokens / 1_000_000) * input_price_per_million
        output_cost = (output_tokens / 1_000_000) * output_price_per_million
        return input_cost + output_cost

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.workflow_tracking = WorkflowTracking(
            workflow_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
            start_time=datetime.now().isoformat()
        )
        
    def track_agent_execution(self, agent_name: str):
        """에이전트 실행을 추적하는 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_datetime = datetime.now()
                
                logger.info(f"🏁 [{agent_name}] 실행 시작: {start_datetime.strftime('%H:%M:%S')}")
                
                execution = AgentExecution(
                    agent_name=agent_name,
                    start_time=start_datetime.isoformat(),
                    end_time="",
                    duration_seconds=0.0
                )
                
                # 컨텍스트 변수에 현재 실행 정보 설정 (LLM 콜백에서 토큰 누적용)
                token1 = _current_agent_name.set(agent_name)
                token2 = _current_execution.set(execution)
                
                try:
                    # 함수 실행
                    result = func(*args, **kwargs)
                    execution.success = True
                    
                    # 결과에서 토큰 정보 추출 시도 (일부 에이전트가 결과에 usage를 넣은 경우 대비)
                    if hasattr(result, 'get') and isinstance(result, dict):
                        self._extract_token_info(result, execution)
                    
                except Exception as e:
                    execution.success = False
                    execution.error_message = str(e)
                    logger.error(f"❌ [{agent_name}] 실행 실패: {str(e)}")
                    raise
                    
                finally:
                    # 컨텍스트 변수 원복
                    try:
                        _current_agent_name.reset(token1)
                        _current_execution.reset(token2)
                    except Exception:
                        pass
                    
                    end_time = time.time()
                    end_datetime = datetime.now()
                    execution.end_time = end_datetime.isoformat()
                    execution.duration_seconds = end_time - start_time
                    
                    logger.info(f"🏁 [{agent_name}] 실행 완료: {execution.duration_seconds:.2f}초")
                    if execution.total_tokens > 0:
                        logger.info(f"💰 [{agent_name}] 토큰 사용: {execution.total_tokens}개 (입력: {execution.input_tokens}, 출력: {execution.output_tokens})")
                        if execution.token_cost_usd > 0:
                            logger.info(f"💳 [{agent_name}] 예상 비용: ${execution.token_cost_usd:.4f}")
                    
                    # 추적 정보 저장
                    self.workflow_tracking.agent_executions.append(execution)
                    self._update_workflow_totals()
                    self.save_tracking_info()
                    
                return result
            return wrapper
        return decorator
    
    def _extract_token_info(self, result: Dict[str, Any], execution: AgentExecution):
        """결과에서 토큰 정보를 추출"""
        token_info = None
        possible_paths = [
            result.get('usage'),
            result.get('token_usage'),
            result.get('llm_output', {}).get('token_usage') if result.get('llm_output') else None,
            result.get('response_metadata', {}).get('token_usage') if result.get('response_metadata') else None
        ]
        for path in possible_paths:
            if path and isinstance(path, dict):
                token_info = path
                break
        if token_info:
            execution.input_tokens = token_info.get('prompt_tokens', 0)
            execution.output_tokens = token_info.get('completion_tokens', 0)
            execution.total_tokens = token_info.get('total_tokens', execution.input_tokens + execution.output_tokens)
            # 모델명 가져와 비용 계산
            model_name = _current_model_name.get()
            execution.token_cost_usd = self._calculate_token_cost(execution.input_tokens, execution.output_tokens, model_name)
    
    def _update_workflow_totals(self):
        """워크플로우 전체 토큰 및 비용 집계"""
        self.workflow_tracking.total_input_tokens = sum(
            exec.input_tokens for exec in self.workflow_tracking.agent_executions
        )
        self.workflow_tracking.total_output_tokens = sum(
            exec.output_tokens for exec in self.workflow_tracking.agent_executions
        )
        self.workflow_tracking.total_tokens = sum(
            exec.total_tokens for exec in self.workflow_tracking.agent_executions
        )
        self.workflow_tracking.total_cost_usd = sum(
            exec.token_cost_usd for exec in self.workflow_tracking.agent_executions
        )
    
    def finalize_workflow(self):
        """워크플로우 종료 시 호출"""
        end_datetime = datetime.now()
        self.workflow_tracking.end_time = end_datetime.isoformat()
        
        start_time = datetime.fromisoformat(self.workflow_tracking.start_time)
        self.workflow_tracking.total_duration_seconds = (end_datetime - start_time).total_seconds()
        
        logger.info(f"🎯 전체 워크플로우 완료")
        logger.info(f"⏱️  총 실행 시간: {self.workflow_tracking.total_duration_seconds:.2f}초")
        logger.info(f"💰 총 토큰 사용: {self.workflow_tracking.total_tokens}개")
        logger.info(f"💳 총 예상 비용: ${self.workflow_tracking.total_cost_usd:.4f}")
        
        self.save_tracking_info()
    
    def save_tracking_info(self):
        """추적 정보를 JSON 파일로 저장"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tracking_file = os.path.join(self.cache_dir, "execution_tracking.json")
            
            with open(tracking_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.workflow_tracking), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"추적 정보 저장 실패: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """추적 정보 요약 반환"""
        return {
            "workflow_id": self.workflow_tracking.workflow_id,
            "total_duration_seconds": self.workflow_tracking.total_duration_seconds,
            "total_agents": len(self.workflow_tracking.agent_executions),
            "successful_agents": sum(1 for exec in self.workflow_tracking.agent_executions if exec.success),
            "failed_agents": sum(1 for exec in self.workflow_tracking.agent_executions if not exec.success),
            "total_tokens": self.workflow_tracking.total_tokens,
            "total_cost_usd": self.workflow_tracking.total_cost_usd,
            "agent_details": [
                {
                    "name": exec.agent_name,
                    "duration": exec.duration_seconds,
                    "tokens": exec.total_tokens,
                    "cost": exec.token_cost_usd,
                    "success": exec.success
                }
                for exec in self.workflow_tracking.agent_executions
            ]
        }


# 전역 추적기 (각 워크플로우 실행 시 초기화)
_global_tracker: Optional[ExecutionTracker] = None


def init_tracking(cache_dir: str):
    """추적 시스템 초기화"""
    global _global_tracker
    _global_tracker = ExecutionTracker(cache_dir)
    logger.info(f"📊 실행 추적 시스템 초기화: {cache_dir}")


def track_agent(agent_name: str):
    """에이전트 추적 데코레이터 - 호출 시점에 추적기를 확인"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _global_tracker:
                return _global_tracker.track_agent_execution(agent_name)(func)(*args, **kwargs)
            # 추적기 미설정 시에도 함수는 정상 실행
            return func(*args, **kwargs)
        return wrapper
    return decorator


def finalize_tracking():
    """추적 종료"""
    global _global_tracker
    if _global_tracker:
        _global_tracker.finalize_workflow()
        return _global_tracker.get_summary()
    return None


def get_tracker() -> Optional[ExecutionTracker]:
    """현재 추적기 반환"""
    return _global_tracker


# ---- 콜백/외부에서 사용될 토큰 집계 헬퍼 ----

def get_current_agent_name() -> Optional[str]:
    return _current_agent_name.get()


def get_current_execution() -> Optional[AgentExecution]:
    return _current_execution.get()


def add_token_usage(input_tokens: int, output_tokens: int):
    """현재 실행 컨텍스트에 토큰 사용량을 누적"""
    exec_obj = _current_execution.get()
    if not exec_obj:
        return
    exec_obj.input_tokens += int(input_tokens or 0)
    exec_obj.output_tokens += int(output_tokens or 0)
    exec_obj.total_tokens = exec_obj.input_tokens + exec_obj.output_tokens
    model_name = _current_model_name.get()
    exec_obj.token_cost_usd = ExecutionTracker._calculate_token_cost(
        input_tokens=exec_obj.input_tokens,
        output_tokens=exec_obj.output_tokens,
        model_name=model_name,
    )


# ---- LangChain 콜백 핸들러 ----
class TokenUsageCallback(BaseCallbackHandler):
    """LLM 호출에서 토큰 사용량을 수집하는 콜백 핸들러"""
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):  # type: ignore[override]
        # 모델명 추출 시도
        model = None
        try:
            if isinstance(serialized, dict):
                # LC 0.3 serialized 구조에서 kwargs.model 또는 model/name
                model = (serialized.get('kwargs') or {}).get('model') or serialized.get('model') or serialized.get('name')
        except Exception:
            pass
        if not model:
            try:
                inv = (kwargs.get('invocation_params') or {})
                model = inv.get('model')
            except Exception:
                pass
        _current_model_name.set(model)

    def on_llm_end(self, output, *, run_id, parent_run_id=None, **kwargs):  # type: ignore[override]
        try:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            llm_output = None
            if hasattr(output, 'llm_output'):
                llm_output = getattr(output, 'llm_output')
                if isinstance(llm_output, dict):
                    usage = llm_output.get('token_usage') or llm_output.get('usage')
                    if isinstance(usage, dict):
                        prompt_tokens = int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
                        completion_tokens = int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
                        total_tokens = int(usage.get('total_tokens') or (prompt_tokens + completion_tokens))
            if (prompt_tokens + completion_tokens) == 0 and hasattr(output, 'generations'):
                try:
                    gens = getattr(output, 'generations') or []
                    for gen_list in gens:
                        for gen in gen_list:
                            msg = getattr(gen, 'message', None)
                            if msg and hasattr(msg, 'response_metadata'):
                                md = getattr(msg, 'response_metadata') or {}
                                usage = md.get('token_usage') or md.get('usage')
                                if isinstance(usage, dict):
                                    prompt_tokens += int(usage.get('prompt_tokens') or usage.get('input_tokens') or 0)
                                    completion_tokens += int(usage.get('completion_tokens') or usage.get('output_tokens') or 0)
                    if total_tokens == 0:
                        total_tokens = prompt_tokens + completion_tokens
                except Exception:
                    pass
            if (prompt_tokens + completion_tokens + total_tokens) > 0:
                # 누적 + 비용 반영
                exec_obj = _current_execution.get()
                if exec_obj:
                    exec_obj.input_tokens += prompt_tokens
                    exec_obj.output_tokens += completion_tokens
                    exec_obj.total_tokens = exec_obj.input_tokens + exec_obj.output_tokens
                    model_name = _current_model_name.get()
                    exec_obj.token_cost_usd = ExecutionTracker._calculate_token_cost(exec_obj.input_tokens, exec_obj.output_tokens, model_name)
        except Exception:
            logging.getLogger(__name__).debug("TokenUsageCallback parsing failed", exc_info=True)


def get_langchain_callback() -> Optional[BaseCallbackHandler]:  # type: ignore[name-defined]
    """LangChain 콜백 인스턴스를 반환 (LangChain이 없으면 None)"""
    try:
        return TokenUsageCallback()
    except Exception:
        return None
