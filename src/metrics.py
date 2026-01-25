"""Evaluation Metrics Module.

Metrics for evaluating interference detection and mitigation.

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_ids(
    attention_divergence: float,
    semantic_conflict: float,
    response_inconsistency: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Compute Interference Detection Score (IDS).
    
    Args:
        attention_divergence: Attention divergence score [0, 1]
        semantic_conflict: Semantic conflict score [0, 1]
        response_inconsistency: Response inconsistency score [0, 1]
        weights: Component weights (default: 0.4, 0.35, 0.25)
        
    Returns:
        IDS score [0, 1]
    """
    if weights is None:
        weights = {
            'attention_divergence': 0.40,
            'semantic_conflict': 0.35,
            'response_inconsistency': 0.25
        }
    
    ids = (
        weights['attention_divergence'] * attention_divergence +
        weights['semantic_conflict'] * semantic_conflict +
        weights['response_inconsistency'] * response_inconsistency
    )
    
    return float(np.clip(ids, 0, 1))


def evaluate_detection(
    predicted_interfering: List[int],
    true_interfering: List[int],
    total_turns: int
) -> Dict[str, float]:
    """Evaluate interference detection accuracy.
    
    Args:
        predicted_interfering: Predicted interfering turn indices
        true_interfering: Ground truth interfering turn indices
        total_turns: Total number of conversation turns
        
    Returns:
        Dictionary with precision, recall, F1, accuracy
    """
    predicted_set = set(predicted_interfering)
    true_set = set(true_interfering)
    
    # True positives, false positives, false negatives
    tp = len(predicted_set & true_set)
    fp = len(predicted_set - true_set)
    fn = len(true_set - predicted_set)
    tn = total_turns - tp - fp - fn
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total_turns if total_turns > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def compute_response_quality(
    response: str,
    ground_truth: str,
    embedding_fn: callable
) -> Dict[str, float]:
    """Compute response quality metrics.
    
    Args:
        response: Generated response
        ground_truth: Expected response
        embedding_fn: Function to compute embeddings
        
    Returns:
        Dictionary with semantic similarity and length ratio
    """
    # Semantic similarity
    emb_response = embedding_fn(response)
    emb_gt = embedding_fn(ground_truth)
    
    similarity = float(
        np.dot(
            emb_response / (np.linalg.norm(emb_response) + 1e-8),
            emb_gt / (np.linalg.norm(emb_gt) + 1e-8)
        )
    )
    
    # Length ratio
    len_response = len(response.split())
    len_gt = len(ground_truth.split())
    length_ratio = len_response / len_gt if len_gt > 0 else 1.0
    
    return {
        'semantic_similarity': similarity,
        'length_ratio': length_ratio,
        'response_length': len_response
    }


def compute_mitigation_effectiveness(
    accuracy_before: float,
    accuracy_after: float,
    context_tokens_before: int,
    context_tokens_after: int
) -> Dict[str, float]:
    """Compute mitigation effectiveness metrics.
    
    Args:
        accuracy_before: Accuracy before mitigation
        accuracy_after: Accuracy after mitigation
        context_tokens_before: Context size before mitigation
        context_tokens_after: Context size after mitigation
        
    Returns:
        Dictionary with improvement metrics
    """
    accuracy_improvement = accuracy_after - accuracy_before
    context_reduction = 1.0 - (context_tokens_after / context_tokens_before) if context_tokens_before > 0 else 0.0
    
    # Efficiency: accuracy per token
    efficiency_before = accuracy_before / context_tokens_before if context_tokens_before > 0 else 0.0
    efficiency_after = accuracy_after / context_tokens_after if context_tokens_after > 0 else 0.0
    efficiency_gain = efficiency_after / efficiency_before if efficiency_before > 0 else 1.0
    
    return {
        'accuracy_improvement': accuracy_improvement,
        'context_reduction': context_reduction,
        'efficiency_gain': efficiency_gain,
        'accuracy_before': accuracy_before,
        'accuracy_after': accuracy_after
    }


def compute_batch_metrics(
    results: List[Dict]
) -> Dict[str, float]:
    """Compute aggregate metrics across multiple experiments.
    
    Args:
        results: List of per-experiment result dictionaries
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not results:
        return {}
    
    # Collect all numeric metrics
    all_metrics = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
    
    # Compute statistics
    batch_metrics = {}
    for metric_name, values in all_metrics.items():
        batch_metrics[f"{metric_name}_mean"] = float(np.mean(values))
        batch_metrics[f"{metric_name}_std"] = float(np.std(values))
        batch_metrics[f"{metric_name}_min"] = float(np.min(values))
        batch_metrics[f"{metric_name}_max"] = float(np.max(values))
    
    return batch_metrics
