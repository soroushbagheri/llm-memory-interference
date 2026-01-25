"""LLM Memory Interference Forensics.

A system for detecting and mitigating context interference in large language models.

Author: Soroush Bagheri
Date: January 2026
"""

__version__ = "0.1.0"
__author__ = "Soroush Bagheri"

from src.detector import InterferenceDetector
from src.tracer import CausalTracer
from src.mitigator import InterferenceMitigator
from src.llm_adapter import LLMAdapter
from src.metrics import compute_ids, evaluate_detection

__all__ = [
    "InterferenceDetector",
    "CausalTracer",
    "InterferenceMitigator",
    "LLMAdapter",
    "compute_ids",
    "evaluate_detection",
]
