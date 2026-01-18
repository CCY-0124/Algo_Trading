"""
log_config.py

Centralized logging configuration with verbosity control.
"""

import logging
import sys
from typing import Optional

# Global verbosity level
_verbosity_level = 1  # 0=quiet, 1=normal, 2=verbose, 3=debug


def set_verbosity(level: int):
    """
    Set global verbosity level.
    
    :param level: 0=quiet (only errors), 1=normal (default), 2=verbose, 3=debug
    """
    global _verbosity_level
    _verbosity_level = max(0, min(3, level))


def get_verbosity() -> int:
    """Get current verbosity level."""
    return _verbosity_level


def should_log(level: str) -> bool:
    """
    Check if message at given level should be logged.
    
    :param level: 'debug', 'info', 'warning', 'error'
    :return: True if should log
    """
    verbosity_map = {
        0: ['error'],           # Quiet: only errors
        1: ['error', 'warning', 'info'],  # Normal: errors, warnings, info
        2: ['error', 'warning', 'info', 'debug'],  # Verbose: all
        3: ['error', 'warning', 'info', 'debug']   # Debug: all
    }
    
    return level in verbosity_map.get(_verbosity_level, verbosity_map[1])


def configure_logging(log_file: Optional[str] = None, verbosity: int = 1, use_progress: bool = True):
    """
    Configure logging with verbosity control.
    
    :param log_file: Optional log file path
    :param verbosity: 0=quiet, 1=normal, 2=verbose, 3=debug
    :param use_progress: Whether to use progress bars (reduces log output)
    """
    set_verbosity(verbosity)
    
    # Determine log level based on verbosity
    level_map = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG
    }
    
    log_level = level_map.get(verbosity, logging.INFO)
    
    # Configure handlers
    handlers = []
    
    # Console handler (with reduced output for normal mode)
    if verbosity >= 1:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        handlers.append(console_handler)
    
    # File handler (always log everything)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG so handlers can filter
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Suppress verbose loggers
    if verbosity < 2:
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


class QuietLogger:
    """
    Context manager to temporarily suppress logging.
    """
    
    def __init__(self, level: int = logging.WARNING):
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        self.original_level = logging.root.level
        logging.root.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.root.setLevel(self.original_level)
