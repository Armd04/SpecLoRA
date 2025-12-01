"""
Tests for the data collector module.
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collector import TrainingExample, DataCollector, AcceptanceRateTracker


class TestTrainingExample:
    """Tests for TrainingExample dataclass."""
    
    def test_create_example(self):
        """Test creating a training example."""
        example = TrainingExample(
            id="test_1",
            prompt="What is Python?",
            draft_output=[1, 2, 3, 4],
            target_output=[1, 2, 5, 6],
            acceptance_rate=0.5,
            timestamp="2024-01-01T00:00:00",
            is_failure=True,
        )
        
        assert example.id == "test_1"
        assert example.prompt == "What is Python?"
        assert example.acceptance_rate == 0.5
        assert example.is_failure is True
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        example = TrainingExample(
            id="test_1",
            prompt="Test prompt",
            draft_output=[1, 2],
            target_output=[1, 3],
            acceptance_rate=0.5,
            timestamp="2024-01-01",
        )
        
        d = example.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test_1"
        assert d["prompt"] == "Test prompt"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "test_2",
            "prompt": "Another prompt",
            "draft_output": [4, 5, 6],
            "target_output": [4, 5, 7],
            "acceptance_rate": 0.66,
            "timestamp": "2024-01-02",
            "is_failure": False,
            "metadata": None,
        }
        
        example = TrainingExample.from_dict(data)
        assert example.id == "test_2"
        assert example.acceptance_rate == 0.66


class TestDataCollector:
    """Tests for DataCollector class."""
    
    def test_init(self):
        """Test initializing the collector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                failures_dir=tmpdir,
                max_failure_cases=10,
                replay_buffer_size=5,
            )
            
            assert collector.max_failure_cases == 10
            assert collector.replay_buffer_size == 5
            assert len(collector.failure_cases) == 0
    
    def test_generate_id(self):
        """Test ID generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(failures_dir=tmpdir)
            
            id1 = collector._generate_id()
            id2 = collector._generate_id()
            
            assert id1 != id2
            assert "example_" in id1
    
    def test_get_training_batch(self):
        """Test getting a training batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                failures_dir=tmpdir,
                max_failure_cases=100,
                replay_buffer_size=10,
                replay_ratio=0.2,
            )
            
            # Add some failure cases
            for i in range(20):
                example = TrainingExample(
                    id=f"failure_{i}",
                    prompt=f"Prompt {i}",
                    draft_output=[i],
                    target_output=[i + 1],
                    acceptance_rate=0.3,
                    timestamp="2024-01-01",
                    is_failure=True,
                )
                collector.failure_cases.append(example)
            
            # Add some replay cases
            for i in range(5):
                example = TrainingExample(
                    id=f"replay_{i}",
                    prompt=f"Replay {i}",
                    draft_output=[i],
                    target_output=[i],
                    acceptance_rate=0.8,
                    timestamp="2024-01-01",
                    is_failure=False,
                )
                collector.replay_buffer.append(example)
            
            # Get batch
            batch = collector.get_training_batch(10)
            
            assert len(batch) == 10
            # Should have some replay cases (roughly 20%)
            replay_count = sum(1 for ex in batch if not ex.is_failure)
            assert replay_count >= 1  # At least some replay cases
    
    def test_stats(self):
        """Test statistics calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(failures_dir=tmpdir, max_failure_cases=50)
            
            # Add failures
            for i in range(10):
                example = TrainingExample(
                    id=f"example_{i}",
                    prompt=f"Prompt {i}",
                    draft_output=[i],
                    target_output=[i + 1],
                    acceptance_rate=0.3 + i * 0.02,
                    timestamp="2024-01-01",
                )
                collector.failure_cases.append(example)
            
            stats = collector.get_stats()
            
            assert stats["num_failure_cases"] == 10
            assert stats["max_failure_cases"] == 50
            assert "avg_failure_acceptance" in stats


class TestAcceptanceRateTracker:
    """Tests for AcceptanceRateTracker class."""
    
    def test_add_rate(self):
        """Test adding acceptance rates."""
        tracker = AcceptanceRateTracker(window_size=10)
        
        tracker.add_rate(0.5)
        tracker.add_rate(0.6)
        tracker.add_rate(0.7)
        
        assert len(tracker.all_rates) == 3
    
    def test_recent_average(self):
        """Test recent average calculation."""
        tracker = AcceptanceRateTracker(window_size=3)
        
        for rate in [0.4, 0.5, 0.6, 0.7, 0.8]:
            tracker.add_rate(rate)
        
        # Recent window is [0.6, 0.7, 0.8]
        avg = tracker.get_recent_average()
        assert abs(avg - 0.7) < 0.01
    
    def test_overall_average(self):
        """Test overall average calculation."""
        tracker = AcceptanceRateTracker()
        
        for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
            tracker.add_rate(rate)
        
        avg = tracker.get_overall_average()
        assert abs(avg - 0.6) < 0.01
    
    def test_trend(self):
        """Test trend calculation."""
        tracker = AcceptanceRateTracker()
        
        # Increasing trend
        for i in range(100):
            tracker.add_rate(0.3 + i * 0.005)
        
        trend = tracker.get_trend(window=25)
        assert trend > 0  # Should be positive (improving)
    
    def test_save_load(self):
        """Test saving and loading tracker data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "tracker.json"
            
            # Create and save
            tracker1 = AcceptanceRateTracker()
            for rate in [0.5, 0.6, 0.7]:
                tracker1.add_rate(rate)
            tracker1.save(str(filepath))
            
            # Load into new tracker
            tracker2 = AcceptanceRateTracker()
            tracker2.load(str(filepath))
            
            assert tracker2.all_rates == tracker1.all_rates
    
    def test_category_tracking(self):
        """Test tracking by category."""
        tracker = AcceptanceRateTracker()
        
        tracker.add_rate(0.5, category="qa")
        tracker.add_rate(0.6, category="qa")
        tracker.add_rate(0.8, category="coding")
        
        stats = tracker.get_stats()
        
        assert "by_category" in stats
        assert "qa" in stats["by_category"]
        assert "coding" in stats["by_category"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
