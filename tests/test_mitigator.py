"""Unit tests for interference mitigator."""

import pytest
from src.mitigator import InterferenceMitigator


class TestInterferenceMitigator:
    """Test suite for InterferenceMitigator."""
    
    @pytest.fixture
    def mitigator(self):
        return InterferenceMitigator()
    
    def test_initialization(self, mitigator):
        """Test mitigator initializes correctly."""
        assert mitigator is not None
        assert hasattr(mitigator, 'mitigate')
    
    def test_soft_masking(self, mitigator):
        """Test soft masking strategy."""
        context = [
            {"role": "user", "content": "Python snakes"},
            {"role": "assistant", "content": "Constrictor snakes"}
        ]
        interfering_turns = [0, 1]
        
        mitigated = mitigator.mitigate(
            context, 
            interfering_turns, 
            strategy="soft_mask"
        )
        
        assert len(mitigated) <= len(context)
    
    def test_hard_masking(self, mitigator):
        """Test hard masking strategy."""
        context = [
            {"role": "user", "content": "Keep this"},
            {"role": "assistant", "content": "Remove this"},
            {"role": "user", "content": "Keep this too"}
        ]
        interfering_turns = [1]
        
        mitigated = mitigator.mitigate(
            context,
            interfering_turns,
            strategy="hard_mask"
        )
        
        assert len(mitigated) == len(context) - 1
        assert mitigated[0]["content"] == "Keep this"
        assert mitigated[1]["content"] == "Keep this too"
    
    def test_no_mitigation_needed(self, mitigator):
        """Test when no interference detected."""
        context = [{"role": "user", "content": "Clean context"}]
        interfering_turns = []
        
        mitigated = mitigator.mitigate(context, interfering_turns)
        assert mitigated == context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
