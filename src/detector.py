"""Interference Detection Module.

Detects when prior conversation context interferes with current reasoning.

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InterferenceResult:
    """Result of interference detection."""
    ids_score: float  # Interference Detection Score [0, 1]
    attention_divergence: float
    semantic_conflict: float
    response_inconsistency: float
    interfering_turns: List[int]
    interfering_tokens: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ids_score': self.ids_score,
            'attention_divergence': self.attention_divergence,
            'semantic_conflict': self.semantic_conflict,
            'response_inconsistency': self.response_inconsistency,
            'interfering_turns': self.interfering_turns,
            'interfering_tokens': self.interfering_tokens,
            'recommendation': self.recommendation
        }


class InterferenceDetector:
    """Detects context interference in LLM conversations."""
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.5,
        n_samples: int = 5
    ):
        """Initialize detector.
        
        Args:
            weights: Component weights for IDS computation
            threshold: IDS threshold for flagging interference
            n_samples: Number of samples for inconsistency detection
        """
        if weights is None:
            self.weights = {
                'attention_divergence': 0.40,
                'semantic_conflict': 0.35,
                'response_inconsistency': 0.25
            }
        else:
            self.weights = weights
        
        self.threshold = threshold
        self.n_samples = n_samples
        
        logger.info(f"InterferenceDetector initialized with threshold={threshold}")
    
    def detect(
        self,
        context: List[Dict[str, str]],
        query: str,
        llm: Any,
        include_trace: bool = False
    ) -> Tuple[float, InterferenceResult]:
        """Detect interference in conversation context.
        
        Args:
            context: List of conversation turns [{"role": str, "content": str}]
            query: Current user query
            llm: LLM adapter instance
            include_trace: Whether to include detailed tracing
            
        Returns:
            (ids_score, InterferenceResult)
        """
        logger.info(f"Detecting interference for query: '{query[:50]}...'")
        
        # Compute components
        attention_div = self._compute_attention_divergence(context, query, llm)
        semantic_conf = self._compute_semantic_conflict(context, query, llm)
        response_incons = self._compute_response_inconsistency(
            context, query, llm, self.n_samples
        )
        
        # Compute weighted IDS
        ids_score = (
            self.weights['attention_divergence'] * attention_div +
            self.weights['semantic_conflict'] * semantic_conf +
            self.weights['response_inconsistency'] * response_incons
        )
        
        # Identify interfering elements
        interfering_turns = self._identify_interfering_turns(
            context, query, attention_div, semantic_conf
        )
        interfering_tokens = self._identify_interfering_tokens(
            context, interfering_turns
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(ids_score, interfering_turns)
        
        result = InterferenceResult(
            ids_score=ids_score,
            attention_divergence=attention_div,
            semantic_conflict=semantic_conf,
            response_inconsistency=response_incons,
            interfering_turns=interfering_turns,
            interfering_tokens=interfering_tokens,
            recommendation=recommendation
        )
        
        logger.info(f"IDS Score: {ids_score:.3f}, Interfering turns: {len(interfering_turns)}")
        
        return ids_score, result
    
    def _compute_attention_divergence(
        self, context: List[Dict], query: str, llm: Any
    ) -> float:
        """Measure split attention between query and past context.
        
        High divergence indicates model is "distracted" by past context.
        """
        if len(context) == 0:
            return 0.0
        
        # Get embeddings for query and context
        query_emb = llm.get_embedding(query)
        context_texts = [turn['content'] for turn in context]
        context_embs = [llm.get_embedding(text) for text in context_texts]
        
        # Compute attention-like weights
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        attention_weights = []
        for ctx_emb in context_embs:
            ctx_norm = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)
            similarity = np.dot(query_norm, ctx_norm)
            attention_weights.append(max(0, similarity))
        
        # Normalize
        total_weight = sum(attention_weights) + 1e-8
        attention_weights = [w / total_weight for w in attention_weights]
        
        # Divergence = entropy of attention distribution
        # High entropy = attention spread across many past turns = high divergence
        if len(attention_weights) > 0:
            entropy = -sum([w * np.log(w + 1e-8) for w in attention_weights if w > 0])
            max_entropy = np.log(len(attention_weights))
            divergence = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            divergence = 0.0
        
        return float(np.clip(divergence, 0, 1))
    
    def _compute_semantic_conflict(
        self, context: List[Dict], query: str, llm: Any
    ) -> float:
        """Detect contradictory information in context.
        
        Measures semantic similarity between context turns to find conflicts.
        """
        if len(context) < 2:
            return 0.0
        
        # Get embeddings for all context turns
        context_texts = [turn['content'] for turn in context]
        context_embs = [llm.get_embedding(text) for text in context_texts]
        
        # Compute pairwise similarities
        n = len(context_embs)
        similarities = []
        
        for i in range(n):
            for j in range(i + 1, n):
                emb_i = context_embs[i] / (np.linalg.norm(context_embs[i]) + 1e-8)
                emb_j = context_embs[j] / (np.linalg.norm(context_embs[j]) + 1e-8)
                sim = np.dot(emb_i, emb_j)
                similarities.append(sim)
        
        if len(similarities) == 0:
            return 0.0
        
        # Low similarity = potential conflict
        # (assuming same topic should have high similarity)
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        
        # Conflict score: higher when similarities are low and variable
        conflict = (1 - avg_sim) * (1 - min_sim)
        
        return float(np.clip(conflict, 0, 1))
    
    def _compute_response_inconsistency(
        self, context: List[Dict], query: str, llm: Any, n_samples: int
    ) -> float:
        """Measure response variance with/without context.
        
        High inconsistency indicates context is confusing the model.
        """
        # Generate responses with full context
        responses_with_context = []
        for _ in range(n_samples):
            response = llm.generate(context + [{"role": "user", "content": query}])
            responses_with_context.append(response)
        
        # Generate responses without context (only query)
        responses_without_context = []
        for _ in range(n_samples):
            response = llm.generate([{"role": "user", "content": query}])
            responses_without_context.append(response)
        
        # Compute embedding variance within each group
        embs_with = [llm.get_embedding(r) for r in responses_with_context]
        embs_without = [llm.get_embedding(r) for r in responses_without_context]
        
        # Variance within groups
        var_with = np.var([embs_with[i] for i in range(len(embs_with))], axis=0).mean()
        var_without = np.var([embs_without[i] for i in range(len(embs_without))], axis=0).mean()
        
        # Inconsistency = ratio of variances
        # High ratio = context increases variance = interference
        if var_without > 1e-8:
            inconsistency = min(var_with / var_without, 2.0) / 2.0
        else:
            inconsistency = 0.5 if var_with > 1e-8 else 0.0
        
        return float(np.clip(inconsistency, 0, 1))
    
    def _identify_interfering_turns(
        self, context: List[Dict], query: str, attention_div: float, semantic_conf: float
    ) -> List[int]:
        """Identify which conversation turns are causing interference."""
        interfering_turns = []
        
        # Simple heuristic: turns with high attention weight and low semantic relevance
        # In real implementation, would use causal tracing
        
        # For now, flag turns if overall interference is high
        if attention_div > 0.6 or semantic_conf > 0.6:
            # Flag oldest turns (most likely to be irrelevant)
            n_flag = min(len(context) // 2, 3)
            interfering_turns = list(range(n_flag))
        
        return interfering_turns
    
    def _identify_interfering_tokens(
        self, context: List[Dict], interfering_turns: List[int]
    ) -> List[str]:
        """Extract key tokens from interfering turns."""
        interfering_tokens = []
        
        for turn_idx in interfering_turns:
            if turn_idx < len(context):
                content = context[turn_idx]['content']
                # Simple extraction: first 3 nouns/keywords
                tokens = content.split()[:5]
                interfering_tokens.extend(tokens)
        
        # Return unique tokens
        return list(set(interfering_tokens))[:10]
    
    def _generate_recommendation(
        self, ids_score: float, interfering_turns: List[int]
    ) -> str:
        """Generate intervention recommendation."""
        if ids_score < self.threshold:
            return "No intervention needed"
        elif ids_score < 0.7:
            return f"Apply soft masking to conversation turns: {interfering_turns}"
        else:
            return f"Apply hard masking (remove) turns: {interfering_turns}"
