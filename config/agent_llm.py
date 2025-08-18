import json
from pathlib import Path
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)


def load_agent_config():
    """agent_config.json 파일에서 설정을 로드합니다."""
    config_path = Path(__file__).parent / "agent_config.json"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # logger.info(f"✅ Agent 설정 파일 로드 완료: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Agent 설정 파일을 찾을 수 없습니다: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Agent 설정 파일 JSON 파싱 오류: {e}")
        return {}


def get_llm(agent_name: str) -> ChatOpenAI:
    """
    특정 agent의 LLM 인스턴스를 반환합니다.
    """
    config = load_agent_config()
    llm_configs = config.get("llm_models", {})

    # 기본값
    default_kwargs = {"model": "gpt-4.1-mini", "temperature": 0}
    agent_config = llm_configs.get(agent_name, default_kwargs)

    # None 값 필터링
    kwargs = {k: v for k, v in agent_config.items() if v is not None}

    logger.info(f"🔧 {agent_name} LLM 생성: {kwargs}")
    base_llm = ChatOpenAI(**kwargs)

    # 토큰 추적 콜백 연결
    try:
        from src.tracking import get_langchain_callback
        cb = get_langchain_callback()
        if cb is not None:
            return base_llm.with_config(callbacks=[cb])
    except Exception:
        pass
    return base_llm
