"""LLM Adapter Module.

Provides unified interface to different LLM APIs (OpenAI, Anthropic, local models).

Author: Soroush Bagheri
Date: January 2026
"""

import os
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM adapters."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        pass


class OpenAIAdapter(BaseLLMAdapter):
    """Adapter for OpenAI models (GPT-4, GPT-3.5, etc.)."""
    
    def __init__(self, model: str = "gpt-4-turbo", temperature: float = 0.7, max_tokens: int = 1024):
        """Initialize OpenAI adapter.
        
        Args:
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"OpenAIAdapter initialized with model: {model}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[ERROR: {str(e)}]"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI embeddings API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.random.randn(1536)  # Fallback


class AnthropicAdapter(BaseLLMAdapter):
    """Adapter for Anthropic Claude models."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", temperature: float = 0.7, max_tokens: int = 1024):
        """Initialize Anthropic adapter."""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"AnthropicAdapter initialized with model: {model}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using Anthropic API."""
        try:
            # Convert OpenAI-style messages to Anthropic format
            system_message = None
            claude_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    claude_messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                temperature=kwargs.get('temperature', self.temperature),
                system=system_message,
                messages=claude_messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return f"[ERROR: {str(e)}]"
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding (uses OpenAI API as fallback)."""
        # Anthropic doesn't provide embeddings API, use OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except:
            logger.warning("Embedding fallback to random")
            return np.random.randn(1536)


class MockLLMAdapter(BaseLLMAdapter):
    """Mock adapter for testing without API calls."""
    
    def __init__(self, seed: int = 42):
        """Initialize mock adapter."""
        self.rng = np.random.RandomState(seed)
        logger.info("MockLLMAdapter initialized (no API calls)")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate mock response."""
        last_message = messages[-1]['content'] if messages else "query"
        return f"Mock response to: {last_message[:50]}..."
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic mock embedding."""
        # Use hash of text as seed for reproducibility
        seed = sum(ord(c) for c in text) % 10000
        rng = np.random.RandomState(seed)
        return rng.randn(1536)


class LLMAdapter:
    """Main adapter that routes to appropriate backend."""
    
    def __init__(self, model: str = "gpt-4-turbo", **kwargs):
        """Initialize adapter based on model name.
        
        Args:
            model: Model identifier (gpt-4-turbo, claude-3-5-sonnet, mock)
            **kwargs: Additional parameters for specific adapter
        """
        if model.startswith("gpt") or model.startswith("o1"):
            self.adapter = OpenAIAdapter(model=model, **kwargs)
        elif model.startswith("claude"):
            self.adapter = AnthropicAdapter(model=model, **kwargs)
        elif model == "mock":
            self.adapter = MockLLMAdapter(**kwargs)
        else:
            logger.warning(f"Unknown model {model}, defaulting to mock")
            self.adapter = MockLLMAdapter()
        
        self.model = model
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response."""
        return self.adapter.generate(messages, **kwargs)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding."""
        return self.adapter.get_embedding(text)
    
    def __repr__(self) -> str:
        return f"LLMAdapter(model={self.model})"
