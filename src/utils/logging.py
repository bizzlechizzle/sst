"""Logging configuration."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from ..core.config import DEFAULT_LOG_DIR


def setup_logging(
    level: int = logging.INFO,
    log_file: bool = True,
    console: bool = True,
) -> logging.Logger:
    """Configure logging for the application.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Whether to log to file
        console: Whether to log to console
        
    Returns:
        Root logger for the application
    """
    # Create logger
    logger = logging.getLogger('sst')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-5s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = DEFAULT_LOG_DIR / f'sst_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_path}")
    
    # Also capture warnings
    logging.captureWarnings(True)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'sst.{name}')


def set_log_level(level: int):
    """Change log level at runtime.
    
    Args:
        level: New logging level
    """
    logger = logging.getLogger('sst')
    logger.setLevel(level)
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(level)


def get_recent_logs(n: int = 100) -> list[str]:
    """Get the most recent log entries.
    
    Args:
        n: Number of lines to return
        
    Returns:
        List of log lines
    """
    # Find most recent log file
    if not DEFAULT_LOG_DIR.exists():
        return []
    
    log_files = sorted(DEFAULT_LOG_DIR.glob('sst_*.log'), reverse=True)
    if not log_files:
        return []
    
    latest_log = log_files[0]
    
    with open(latest_log, 'r') as f:
        lines = f.readlines()
    
    return lines[-n:]
