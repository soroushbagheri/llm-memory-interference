"""Interference Mitigation Module.

Implements strategies to mitigate detected interference.

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MitigationResult:
    """Result of mitigation intervention."""
    strategy: str
    modified_context: List[Dict[str, str]]
    removed_turns: List[int]
    masked_tokens: List[str]
    metadata: Dict
    

class InterferenceMitigator:
    """Applies intervention strategies to mitigate interference."""
    
    STRATEGIES = ["none", "soft_masking", "hard_masking", "reordering"]
    
    def __init__(self, default_strategy: str = "soft_masking"):
        """Initialize mitigator.
        
        Args:
            default_strategy: Default mitigation strategy
        """
        if default_strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {default_strategy}. Choose from {self.STRATEGIES}")
        
        self.default_strategy = default_strategy
        logger.info(f"InterferenceMitigator initialized with strategy: {default_strategy}")
    
    def mitigate(
        self,
        context: List[Dict[str, str]],
        interfering_turns: List[int],
        strategy: Optional[str] = None,
        **kwargs
    ) -> MitigationResult:
        """Apply mitigation strategy to context.
        
        Args:
            context: Conversation history
            interfering_turns: Indices of interfering turns
            strategy: Mitigation strategy (if None, use default)
            **kwargs: Strategy-specific parameters
            
        Returns:
            MitigationResult with modified context
        """
        if strategy is None:
            strategy = self.default_strategy
        
        logger.info(f"Applying {strategy} to {len(interfering_turns)} interfering turns")
        
        if strategy == "none":
            return self._no_intervention(context)
        elif strategy == "soft_masking":
            return self._soft_masking(context, interfering_turns, **kwargs)
        elif strategy == "hard_masking":
            return self._hard_masking(context, interfering_turns)
        elif strategy == "reordering":
            return self._reordering(context, interfering_turns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _no_intervention(self, context: List[Dict]) -> MitigationResult:
        """No intervention (baseline)."""
        return MitigationResult(
            strategy="none",
            modified_context=context.copy(),
            removed_turns=[],
            masked_tokens=[],
            metadata={"intervention": "none"}
        )
    
    def _soft_masking(
        self,
        context: List[Dict],
        interfering_turns: List[int],
        mask_token: str = "[MASKED]",
        mask_ratio: float = 0.5
    ) -> MitigationResult:
        """Soft masking: replace portions of interfering content.
        
        Args:
            context: Conversation history
            interfering_turns: Turn indices to mask
            mask_token: Token to use for masking
            mask_ratio: Fraction of tokens to mask (0.0-1.0)
        """
        modified_context = context.copy()
        masked_tokens_all = []
        
        for turn_idx in interfering_turns:
            if turn_idx < len(modified_context):
                turn = modified_context[turn_idx]
                tokens = turn['content'].split()
                n_to_mask = int(len(tokens) * mask_ratio)
                
                # Mask first n_to_mask tokens (could use more sophisticated selection)
                masked_tokens = tokens[:n_to_mask]
                masked_tokens_all.extend(masked_tokens)
                
                # Replace with mask token
                tokens[:n_to_mask] = [mask_token] * n_to_mask
                modified_context[turn_idx]['content'] = " ".join(tokens)
        
        return MitigationResult(
            strategy="soft_masking",
            modified_context=modified_context,
            removed_turns=[],
            masked_tokens=masked_tokens_all,
            metadata={
                "mask_token": mask_token,
                "mask_ratio": mask_ratio,
                "n_turns_masked": len(interfering_turns)
            }
        )
    
    def _hard_masking(
        self,
        context: List[Dict],
        interfering_turns: List[int]
    ) -> MitigationResult:
        """Hard masking: completely remove interfering turns."""
        modified_context = [
            turn for i, turn in enumerate(context)
            if i not in interfering_turns
        ]
        
        removed_content = [
            context[i]['content'] for i in interfering_turns if i < len(context)
        ]
        
        return MitigationResult(
            strategy="hard_masking",
            modified_context=modified_context,
            removed_turns=interfering_turns,
            masked_tokens=[],
            metadata={
                "n_turns_removed": len(interfering_turns),
                "removed_content_length": sum(len(c.split()) for c in removed_content)
            }
        )
    
    def _reordering(
        self,
        context: List[Dict],
        interfering_turns: List[int]
    ) -> MitigationResult:
        """Reordering: move interfering turns to end (furthest from query)."""
        # Separate interfering and non-interfering turns
        non_interfering = [
            turn for i, turn in enumerate(context) if i not in interfering_turns
        ]
        interfering = [
            turn for i, turn in enumerate(context) if i in interfering_turns
        ]
        
        # Reorder: non-interfering first, interfering at end
        modified_context = non_interfering + interfering
        
        return MitigationResult(
            strategy="reordering",
            modified_context=modified_context,
            removed_turns=[],
            masked_tokens=[],
            metadata={
                "n_turns_reordered": len(interfering_turns),
                "new_positions": list(range(len(non_interfering), len(context)))
            }
        )
    
    def evaluate_mitigation(
        self,
        original_context: List[Dict],
        mitigated_context: List[Dict],
        query: str,
        llm: any,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate effectiveness of mitigation.
        
        Args:
            original_context: Context before mitigation
            mitigated_context: Context after mitigation
            query: User query
            llm: LLM adapter
            ground_truth: Expected answer (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Generate responses
        response_original = llm.generate(
            original_context + [{"role": "user", "content": query}]
        )
        response_mitigated = llm.generate(
            mitigated_context + [{"role": "user", "content": query}]
        )
        
        # Compute metrics
        metrics = {}
        
        # Context reduction
        orig_tokens = sum(len(t['content'].split()) for t in original_context)
        mit_tokens = sum(len(t['content'].split()) for t in mitigated_context)
        metrics['context_reduction_ratio'] = 1.0 - (mit_tokens / orig_tokens if orig_tokens > 0 else 0)
        
        # Response similarity (lower = more different)
        emb_orig = llm.get_embedding(response_original)
        emb_mit = llm.get_embedding(response_mitigated)
        metrics['response_similarity'] = float(
            np.dot(
                emb_orig / (np.linalg.norm(emb_orig) + 1e-8),
                emb_mit / (np.linalg.norm(emb_mit) + 1e-8)
            )
        )
        
        # Accuracy (if ground truth provided)
        if ground_truth:
            emb_gt = llm.get_embedding(ground_truth)
            
            metrics['accuracy_original'] = float(
                np.dot(
                    emb_orig / (np.linalg.norm(emb_orig) + 1e-8),
                    emb_gt / (np.linalg.norm(emb_gt) + 1e-8)
                )
            )
            metrics['accuracy_mitigated'] = float(
                np.dot(
                    emb_mit / (np.linalg.norm(emb_mit) + 1e-8),
                    emb_gt / (np.linalg.norm(emb_gt) + 1e-8)
                )
            )
            metrics['accuracy_improvement'] = metrics['accuracy_mitigated'] - metrics['accuracy_original']
        
        logger.info(f"Mitigation evaluation: {metrics}")
        return metrics
