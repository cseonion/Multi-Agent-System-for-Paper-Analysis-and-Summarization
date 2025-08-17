import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


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
                
                try:
                    # 함수 실행
                    result = func(*args, **kwargs)
                    execution.success = True
                    
                    # 결과에서 토큰 정보 추출 시도
                    if hasattr(result, 'get') and isinstance(result, dict):
                        # LLM 응답에 토큰 정보가 있는지 확인
                        self._extract_token_info(result, execution)
                    
                except Exception as e:
                    execution.success = False
                    execution.error_message = str(e)
                    logger.error(f"❌ [{agent_name}] 실행 실패: {str(e)}")
                    raise
                    
                finally:
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
        # 일반적인 LLM 응답 구조에서 토큰 정보 찾기
        token_info = None
        
        # 다양한 경로에서 토큰 정보 찾기
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
            execution.total_tokens = token_info.get('total_tokens', 
                                                  execution.input_tokens + execution.output_tokens)
            
            # 간단한 비용 계산 (OpenAI GPT-4 기준 예시)
            # 실제 사용하는 모델에 따라 조정 필요
            execution.token_cost_usd = self._calculate_token_cost(
                execution.input_tokens, 
                execution.output_tokens
            )
    
    def _calculate_token_cost(self, input_tokens: int, output_tokens: int) -> float:
        """토큰 비용 계산 (OpenAI GPT-4 기준 예시)"""
        # GPT-4 가격 (2024년 기준, 실제 사용 모델에 맞게 조정)
        input_cost_per_1k = 0.03  # $0.03 per 1K input tokens
        output_cost_per_1k = 0.06  # $0.06 per 1K output tokens
        
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        
        return input_cost + output_cost
    
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
    """에이전트 추적 데코레이터"""
    def decorator(func):
        if _global_tracker:
            return _global_tracker.track_agent_execution(agent_name)(func)
        else:
            # 추적기가 없으면 원본 함수 반환
            return func
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
