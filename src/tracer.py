"""Causal Tracing Module.

Traces specific tokens causing interference using gradient-based attribution.

Author: Soroush Bagheri
Date: January 2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CausalTracer:
    """Traces causal influence of tokens on model outputs."""
    
    def __init__(self, attribution_method: str = "gradient"):
        """Initialize tracer.
        
        Args:
            attribution_method: Method for attribution ("gradient", "attention", "integrated_gradients")
        """
        self.attribution_method = attribution_method
        logger.info(f"CausalTracer initialized with method: {attribution_method}")
    
    def trace(
        self,
        context: List[Dict[str, str]],
        query: str,
        response: str,
        llm: any,
        top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Trace which tokens from context caused interference.
        
        Args:
            context: Conversation history
            query: Current query
            response: Model's response
            llm: LLM adapter with gradient access
            top_k: Number of top interfering tokens to return
            
        Returns:
            Dictionary mapping turn indices to (token, attribution_score) pairs
        """
        logger.info(f"Tracing causal influence using {self.attribution_method}")
        
        if self.attribution_method == "gradient":
            return self._gradient_attribution(context, query, response, llm, top_k)
        elif self.attribution_method == "attention":
            return self._attention_attribution(context, query, response, llm, top_k)
        else:
            raise ValueError(f"Unknown attribution method: {self.attribution_method}")
    
    def _gradient_attribution(
        self,
        context: List[Dict],
        query: str,
        response: str,
        llm: any,
        top_k: int
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Gradient-based attribution.
        
        Computes ||∇_token Loss(response)|| for each context token.
        """
        # Concatenate all context into single text
        context_text = " ".join([turn['content'] for turn in context])
        tokens = context_text.split()
        
        # Simulate gradient computation (in real implementation, use actual gradients)
        # For now, use embedding-based approximation
        attributions = {}
        
        query_emb = llm.get_embedding(query)
        response_emb = llm.get_embedding(response)
        
        # "Gradient" = how much each token embedding aligns with query-response direction
        direction = response_emb - query_emb
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        token_scores = []
        for i, token in enumerate(tokens):
            token_emb = llm.get_embedding(token)
            token_norm = token_emb / (np.linalg.norm(token_emb) + 1e-8)
            
            # Attribution = projection onto query-response direction
            attribution = abs(np.dot(token_norm, direction))
            token_scores.append((token, float(attribution)))
        
        # Sort by attribution score
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Group by turn
        attributions['all_context'] = token_scores[:top_k]
        
        return attributions
    
    def _attention_attribution(
        self,
        context: List[Dict],
        query: str,
        response: str,
        llm: any,
        top_k: int
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Attention-based attribution.
        
        Uses attention weights from model (requires model access).
        """
        # Placeholder: would use actual attention weights from model
        # For now, use similarity-based proxy
        
        attributions = {}
        response_emb = llm.get_embedding(response)
        
        for turn_idx, turn in enumerate(context):
            tokens = turn['content'].split()
            token_scores = []
            
            for token in tokens:
                token_emb = llm.get_embedding(token)
                
                # Attention proxy: similarity to response
                similarity = np.dot(
                    token_emb / (np.linalg.norm(token_emb) + 1e-8),
                    response_emb / (np.linalg.norm(response_emb) + 1e-8)
                )
                token_scores.append((token, float(similarity)))
            
            # Sort and take top-k
            token_scores.sort(key=lambda x: x[1], reverse=True)
            attributions[f"turn_{turn_idx}"] = token_scores[:min(top_k, len(token_scores))]
        
        return attributions
    
    def visualize_attribution(
        self,
        attributions: Dict[str, List[Tuple[str, float]]],
        save_path: Optional[str] = None
    ) -> str:
        """Generate visualization of attribution scores.
        
        Args:
            attributions: Output from trace()
            save_path: Path to save visualization (optional)
            
        Returns:
            HTML string with visualization
        """
        html = "<div style='font-family: monospace;'>\n"
        html += "<h3>Causal Attribution Scores</h3>\n"
        
        for turn_name, token_scores in attributions.items():
            html += f"<h4>{turn_name}</h4>\n"
            html += "<ul>\n"
            
            for token, score in token_scores:
                # Color code by score (red = high attribution)
                color_intensity = int(score * 255)
                color = f"rgb({color_intensity}, {255 - color_intensity}, 0)"
                html += f"<li><span style='background-color:{color}; padding:2px;'>{token}</span>: {score:.3f}</li>\n"
            
            html += "</ul>\n"
        
        html += "</div>"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(html)
            logger.info(f"Attribution visualization saved to {save_path}")
        
        return html
