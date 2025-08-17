"""
로깅 설정 모듈
환경변수와 설정 파일을 통해 로깅을 중앙 관리
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logging(
    log_level: str = None,
    log_to_file: bool = True,
    log_file_path: str = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
):
    """
    통합 로깅 설정
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: 파일로 로그 저장 여부
        log_file_path: 로그 파일 경로
        max_file_size: 로그 파일 최대 크기 (bytes)
        backup_count: 백업 파일 개수
    """
    
    # 환경변수에서 로그 레벨 가져오기
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # 로그 레벨 설정
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 로그 포맷 설정
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택적)
    if log_to_file:
        if log_file_path is None:
            os.makedirs('logs', exist_ok=True)
            log_file_path = f'logs/task_paper_{datetime.now().strftime("%Y%m%d")}.log'
        
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name: str):
    """모듈별 로거 생성"""
    return logging.getLogger(name)

# 환경변수 기반 자동 설정
def auto_setup():
    """환경변수를 기반으로 자동 로깅 설정"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    log_to_file = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    log_file = os.getenv('LOG_FILE_PATH', None)
    
    setup_logging(
        log_level=log_level,
        log_to_file=log_to_file,
        log_file_path=log_file
    )

# 실행 시 자동 설정
if __name__ != "__main__":
    # 모듈이 import될 때 자동으로 설정
    # auto_setup()  # 필요시 주석 해제
    pass
