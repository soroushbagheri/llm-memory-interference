"""Unit tests for interference detector."""

import pytest
from src.detector import InterferenceDetector
from src.llm_adapter import LLMAdapter


class TestInterferenceDetector:
    """Test suite for InterferenceDetector."""
    
    @pytest.fixture
    def detector(self):
        return InterferenceDetector()
    
    @pytest.fixture
    def llm(self):
        # Mock LLM for testing
        return LLMAdapter(model="mock")
    
    def test_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert hasattr(detector, 'detect')
    
    def test_detect_no_interference(self, detector, llm):
        """Test detection on clean context."""
        context = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."}
        ]
        query = "What is 3+3?"
        
        # Expected: low IDS score
        ids, details = detector.detect(context, query, llm)
        assert 0 <= ids <= 1.0, "IDS should be between 0 and 1"
        assert ids < 0.5, "Clean context should have low IDS"
    
    def test_detect_lexical_interference(self, detector, llm):
        """Test detection of lexical interference."""
        context = [
            {"role": "user", "content": "Tell me about Python snakes."},
            {"role": "assistant", "content": "Pythons are large constrictor snakes."}
        ]
        query = "How do I install Python packages?"
        
        # Expected: high IDS score
        ids, details = detector.detect(context, query, llm)
        assert ids > 0.5, "Lexical interference should have high IDS"
    
    def test_ids_range(self, detector, llm):
        """Test IDS is always in valid range."""
        context = [{"role": "user", "content": "Test"}]
        query = "Query"
        
        ids, _ = detector.detect(context, query, llm)
        assert 0 <= ids <= 1.0, "IDS must be in [0, 1] range"
    
    def test_empty_context(self, detector, llm):
        """Test behavior with empty context."""
        context = []
        query = "What is the capital of France?"
        
        ids, details = detector.detect(context, query, llm)
        assert ids == 0.0, "Empty context should have zero interference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
