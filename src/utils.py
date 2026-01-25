"""Utility Functions.

Helper functions for data processing, file I/O, and visualization.

Author: Soroush Bagheri
Date: January 2026
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def save_json(data: Any, filepath: str, indent: int = 2):
    """Save data to JSON file.
    
    Args:
        data: Data to save (must be JSON serializable)
        filepath: Output file path
        indent: JSON indentation
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
        format_str: Log message format
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )


def ensure_dir(directory: str):
    """Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity [-1, 1]
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def format_conversation(messages: List[Dict[str, str]], max_length: int = 500) -> str:
    """Format conversation for display.
    
    Args:
        messages: List of message dictionaries
        max_length: Maximum length per message
        
    Returns:
        Formatted string
    """
    lines = []
    for msg in messages:
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > max_length:
            content = content[:max_length] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def truncate_context(
    context: List[Dict[str, str]],
    max_turns: Optional[int] = None,
    max_tokens: Optional[int] = None
) -> List[Dict[str, str]]:
    """Truncate conversation context.
    
    Args:
        context: Conversation history
        max_turns: Maximum number of turns to keep
        max_tokens: Maximum total tokens (approximate)
        
    Returns:
        Truncated context
    """
    if max_turns and len(context) > max_turns:
        context = context[-max_turns:]
    
    if max_tokens:
        total_tokens = sum(len(turn['content'].split()) for turn in context)
        while total_tokens > max_tokens and len(context) > 1:
            context.pop(0)
            total_tokens = sum(len(turn['content'].split()) for turn in context)
    
    return context
