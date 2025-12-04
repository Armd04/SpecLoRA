"""
Tests for TokenLevelDisagreement and related manual speculative decoding features.
"""

import json
import tempfile
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collector import (
    TokenLevelDisagreement,
    TrainingExample,
    DataCollector,
)


class TestTokenLevelDisagreement:
    """Tests for TokenLevelDisagreement dataclass."""
    
    def test_create_disagreement(self):
        """Test creating a token-level disagreement."""
        disagreement = TokenLevelDisagreement(
            position=10,
            draft_token=1234,
            target_token=5678,
            draft_confidence=0.85,
            target_confidence=0.95,
            context_tokens=[1, 2, 3, 4, 5],
        )
        
        assert disagreement.position == 10
        assert disagreement.draft_token == 1234
        assert disagreement.target_token == 5678
        assert disagreement.draft_confidence == 0.85
        assert disagreement.target_confidence == 0.95
        assert disagreement.context_tokens == [1, 2, 3, 4, 5]
    
    def test_high_confidence_failure(self):
        """Test detecting high-confidence failures."""
        # High confidence (> 0.4) - should be True
        high_conf = TokenLevelDisagreement(
            position=1,
            draft_token=100,
            target_token=200,
            draft_confidence=0.6,
            target_confidence=0.9,
        )
        assert high_conf.is_high_confidence_failure is True
        
        # Low confidence (< 0.4) - should be False
        low_conf = TokenLevelDisagreement(
            position=1,
            draft_token=100,
            target_token=200,
            draft_confidence=0.3,
            target_confidence=0.9,
        )
        assert low_conf.is_high_confidence_failure is False
        
        # Boundary case (exactly 0.4) - should be False
        boundary = TokenLevelDisagreement(
            position=1,
            draft_token=100,
            target_token=200,
            draft_confidence=0.4,
            target_confidence=0.9,
        )
        assert boundary.is_high_confidence_failure is False
    
    def test_to_dict(self):
        """Test converting disagreement to dictionary."""
        disagreement = TokenLevelDisagreement(
            position=5,
            draft_token=100,
            target_token=200,
            draft_confidence=0.7,
            target_confidence=0.8,
            context_tokens=[10, 20, 30],
        )
        
        d = disagreement.to_dict()
        
        assert isinstance(d, dict)
        assert d["position"] == 5
        assert d["draft_token"] == 100
        assert d["target_token"] == 200
        assert d["draft_confidence"] == 0.7
        assert d["context_tokens"] == [10, 20, 30]
    
    def test_from_dict(self):
        """Test creating disagreement from dictionary."""
        data = {
            "position": 15,
            "draft_token": 300,
            "target_token": 400,
            "draft_confidence": 0.5,
            "target_confidence": 0.6,
            "context_tokens": [1, 2],
        }
        
        disagreement = TokenLevelDisagreement.from_dict(data)
        
        assert disagreement.position == 15
        assert disagreement.draft_token == 300
        assert disagreement.target_token == 400
    
    def test_serialization_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = TokenLevelDisagreement(
            position=7,
            draft_token=111,
            target_token=222,
            draft_confidence=0.55,
            target_confidence=0.66,
            context_tokens=[5, 6, 7, 8],
        )
        
        # Round trip through dict
        d = original.to_dict()
        restored = TokenLevelDisagreement.from_dict(d)
        
        assert restored.position == original.position
        assert restored.draft_token == original.draft_token
        assert restored.target_token == original.target_token
        assert restored.draft_confidence == original.draft_confidence
        assert restored.target_confidence == original.target_confidence
        assert restored.context_tokens == original.context_tokens
    
    def test_json_serialization(self):
        """Test that disagreements can be serialized to JSON."""
        disagreement = TokenLevelDisagreement(
            position=3,
            draft_token=50,
            target_token=60,
            draft_confidence=0.45,
            target_confidence=0.55,
            context_tokens=[1, 2, 3],
        )
        
        # Serialize to JSON string
        json_str = json.dumps(disagreement.to_dict())
        
        # Deserialize back
        data = json.loads(json_str)
        restored = TokenLevelDisagreement.from_dict(data)
        
        assert restored.position == 3
        assert restored.draft_token == 50


class TestTrainingExampleWithDisagreements:
    """Tests for TrainingExample with disagreement data."""
    
    def test_example_without_disagreements(self):
        """Test training example without disagreement data."""
        example = TrainingExample(
            id="test_1",
            prompt="Test prompt",
            draft_output=[1, 2, 3],
            target_output=[1, 2, 4],
            acceptance_rate=0.66,
            timestamp="2024-01-01",
        )
        
        assert example.has_detailed_data is False
        assert example.high_confidence_failures == []
    
    def test_example_with_disagreements(self):
        """Test training example with disagreement data."""
        disagreements = [
            TokenLevelDisagreement(
                position=5,
                draft_token=100,
                target_token=200,
                draft_confidence=0.6,  # High confidence
                target_confidence=0.9,
            ),
            TokenLevelDisagreement(
                position=10,
                draft_token=300,
                target_token=400,
                draft_confidence=0.2,  # Low confidence
                target_confidence=0.8,
            ),
        ]
        
        example = TrainingExample(
            id="test_2",
            prompt="Test prompt",
            draft_output=[1, 2, 3, 4, 5],
            target_output=[1, 2, 3, 4, 6],
            acceptance_rate=0.8,
            timestamp="2024-01-01",
            disagreements=disagreements,
        )
        
        assert example.has_detailed_data is True
        assert len(example.disagreements) == 2
        
        # Only one high-confidence failure
        high_conf = example.high_confidence_failures
        assert len(high_conf) == 1
        assert high_conf[0].position == 5
    
    def test_example_serialization_with_disagreements(self):
        """Test serialization of training examples with disagreements."""
        disagreements = [
            TokenLevelDisagreement(
                position=3,
                draft_token=50,
                target_token=60,
                draft_confidence=0.7,
                target_confidence=0.8,
                context_tokens=[1, 2],
            ),
        ]
        
        example = TrainingExample(
            id="test_3",
            prompt="Serialize test",
            draft_output=[10, 20, 30],
            target_output=[10, 20, 40],
            acceptance_rate=0.66,
            timestamp="2024-01-01",
            disagreements=disagreements,
        )
        
        # Serialize
        d = example.to_dict()
        
        assert "disagreements" in d
        assert len(d["disagreements"]) == 1
        assert d["disagreements"][0]["position"] == 3
        
        # Deserialize
        restored = TrainingExample.from_dict(d)
        
        assert restored.has_detailed_data is True
        assert len(restored.disagreements) == 1
        assert restored.disagreements[0].position == 3
        assert restored.disagreements[0].draft_token == 50


class TestDataCollectorWithDetailedData:
    """Tests for DataCollector with detailed disagreement data."""
    
    def test_add_detailed_result(self):
        """Test adding a result with detailed disagreement data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                failures_dir=tmpdir,
                max_failure_cases=100,
            )
            
            disagreements = [
                TokenLevelDisagreement(
                    position=5,
                    draft_token=100,
                    target_token=200,
                    draft_confidence=0.5,
                    target_confidence=0.8,
                ),
            ]
            
            # Add detailed result
            should_train = collector.add_detailed_result(
                prompt="Test prompt",
                generated_tokens=[1, 2, 3, 4, 5],
                disagreements=disagreements,
                acceptance_rate=0.6,
            )
            
            assert should_train is False  # Not enough cases yet
            assert len(collector.failure_cases) == 1
            
            # Check the stored example
            example = collector.failure_cases[0]
            assert example.has_detailed_data is True
            assert len(example.disagreements) == 1
    
    def test_export_includes_disagreements(self):
        """Test that export includes disagreement data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(failures_dir=tmpdir)
            
            disagreements = [
                TokenLevelDisagreement(
                    position=3,
                    draft_token=50,
                    target_token=60,
                    draft_confidence=0.7,
                    target_confidence=0.8,
                    context_tokens=[1, 2],
                ),
            ]
            
            collector.add_detailed_result(
                prompt="Export test",
                generated_tokens=[1, 2, 3],
                disagreements=disagreements,
                acceptance_rate=0.5,
            )
            
            # Export
            output_path = Path(tmpdir) / "export.jsonl"
            collector.export_for_training(str(output_path))
            
            # Read and verify
            with open(output_path) as f:
                line = f.readline()
                data = json.loads(line)
            
            assert "disagreements" in data
            assert "disagreement_positions" in data
            assert data["disagreement_positions"] == [3]
            assert "high_confidence_failure_positions" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
