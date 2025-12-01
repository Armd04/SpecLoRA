"""
Utility functions for speculative decoding system.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger("speculative_lora")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_memory(bytes_count: int) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format seconds as human-readable string.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def count_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """
    Estimate token count from text (rough approximation).
    
    Args:
        text: Input text
        chars_per_token: Average characters per token
        
    Returns:
        Estimated token count
    """
    return int(len(text) / chars_per_token)


class MovingAverage:
    """
    Simple moving average calculator.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []
    
    def add(self, value: float) -> None:
        """Add a value to the moving average."""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
    
    def get(self) -> float:
        """Get the current moving average."""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)
    
    def __len__(self) -> int:
        return len(self.values)


class ProgressTracker:
    """
    Track progress of long-running operations.
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        self.current = min(self.current + n, self.total)
    
    @property
    def progress(self) -> float:
        """Get progress as fraction (0-1)."""
        if self.total == 0:
            return 1.0
        return self.current / self.total
    
    @property
    def percent(self) -> float:
        """Get progress as percentage."""
        return self.progress * 100
    
    def __str__(self) -> str:
        return f"{self.description}: {self.current}/{self.total} ({self.percent:.1f}%)"
