"""
Tests for configurable loss functions (cross-entropy, KL divergence, mixed).
"""

import mlx.core as mx
import mlx.nn as nn

from src.training import (
    LossConfig,
    cross_entropy_loss,
    kl_divergence_loss,
    mixed_loss,
    LoRATrainer,
    LoRAConfig,
)
from src.data_collector import TokenLevelDisagreement


class TestLossConfig:
    """Test LossConfig dataclass."""

    def test_default_config(self):
        """Test default LossConfig values."""
        config = LossConfig()
        assert config.type == "cross_entropy"
        assert config.temperature == 2.0
        assert config.alpha == 0.5
        assert config.top_k_logits == 10

    def test_custom_config(self):
        """Test custom LossConfig values."""
        config = LossConfig(
            type="kl_divergence", temperature=3.0, alpha=0.7, top_k_logits=5
        )
        assert config.type == "kl_divergence"
        assert config.temperature == 3.0
        assert config.alpha == 0.7
        assert config.top_k_logits == 5


class TestCrossEntropyLoss:
    """Test cross-entropy loss function."""

    def test_basic_ce_loss(self):
        """Test basic cross-entropy loss computation."""
        vocab_size = 1000
        batch_size = 2
        seq_len = 5

        # Create dummy logits and targets
        logits = mx.random.normal((batch_size * seq_len, vocab_size))
        targets = mx.array([10, 20, 30, 40, 50, 15, 25, 35, 45, 55])
        mask = mx.array([True] * (batch_size * seq_len))

        loss = cross_entropy_loss(logits, targets, mask)
        mx.eval(loss)

        assert loss.shape == ()
        assert mx.isfinite(loss).item()
        assert loss.item() > 0

    def test_ce_loss_with_masking(self):
        """Test cross-entropy loss with masked positions."""
        vocab_size = 100
        logits = mx.random.normal((10, vocab_size))
        targets = mx.array([1, 2, -100, 4, 5, -100, 7, 8, 9, 10])
        mask = targets != -100

        loss = cross_entropy_loss(logits, targets, mask)
        mx.eval(loss)

        assert loss.shape == ()
        assert mx.isfinite(loss).item()
        assert loss.item() > 0

    def test_ce_loss_all_masked(self):
        """Test cross-entropy loss with all positions masked."""
        vocab_size = 100
        logits = mx.random.normal((5, vocab_size))
        targets = mx.array([-100, -100, -100, -100, -100])
        mask = targets != -100

        loss = cross_entropy_loss(logits, targets, mask)
        mx.eval(loss)

        assert mx.isnan(loss).item()


class TestKLDivergenceLoss:
    """Test KL divergence loss function."""

    def test_basic_kl_loss(self):
        """Test basic KL divergence loss computation."""
        vocab_size = 100
        batch_size = 2
        seq_len = 3

        # Create student logits
        student_logits = mx.random.normal((batch_size * seq_len, vocab_size))

        # Create sparse target logits (top-k only)
        # Format: Dict[flat_idx -> Dict[token_id -> prob]]
        target_logits_sparse = {
            0: {10: 0.3, 20: 0.2, 30: 0.1},  # Position 0: top-3 tokens
            1: {15: 0.4, 25: 0.3},  # Position 1: top-2 tokens
            5: {40: 0.5, 50: 0.3},  # Position 5: top-2 tokens
        }

        # Create position mask (MLX arrays are immutable, so create with values)
        # Positions 0, 1, and 5 should be True
        positions_list = [False] * (batch_size * seq_len)
        positions_list[0] = True
        positions_list[1] = True
        positions_list[5] = True
        positions = mx.array(positions_list)

        temperature = 2.0
        loss = kl_divergence_loss(
            student_logits, target_logits_sparse, positions, temperature, vocab_size
        )
        mx.eval(loss)

        assert loss.shape == ()
        assert mx.isfinite(loss).item()
        assert loss.item() >= 0  # KL divergence is non-negative

    def test_kl_loss_no_positions(self):
        """Test KL loss with no valid positions."""
        vocab_size = 100
        student_logits = mx.random.normal((10, vocab_size))
        target_logits_sparse = {}
        positions = mx.array([False] * 10)
        temperature = 2.0

        loss = kl_divergence_loss(
            student_logits, target_logits_sparse, positions, temperature, vocab_size
        )
        mx.eval(loss)

        assert mx.isnan(loss).item()


class TestMixedLoss:
    """Test mixed loss function."""

    def test_mixed_loss(self):
        """Test mixed loss combining CE and KL."""
        ce_loss = mx.array(2.5)
        kl_loss = mx.array(1.5)
        alpha = 0.5

        mixed = mixed_loss(ce_loss, kl_loss, alpha)
        mx.eval(mixed)

        expected = 0.5 * 2.5 + 0.5 * 1.5
        assert abs(mixed.item() - expected) < 1e-5

    def test_mixed_loss_alpha_zero(self):
        """Test mixed loss with alpha=0 (pure KL)."""
        ce_loss = mx.array(2.0)
        kl_loss = mx.array(1.0)
        alpha = 0.0

        mixed = mixed_loss(ce_loss, kl_loss, alpha)
        mx.eval(mixed)

        assert abs(mixed.item() - 1.0) < 1e-5

    def test_mixed_loss_alpha_one(self):
        """Test mixed loss with alpha=1 (pure CE)."""
        ce_loss = mx.array(2.0)
        kl_loss = mx.array(1.0)
        alpha = 1.0

        mixed = mixed_loss(ce_loss, kl_loss, alpha)
        mx.eval(mixed)

        assert abs(mixed.item() - 2.0) < 1e-5


class TestLoRATrainerLossIntegration:
    """Test LoRATrainer with different loss configurations."""

    def test_trainer_with_cross_entropy(self):
        """Test LoRATrainer with cross-entropy loss (default)."""
        # Create a minimal model for testing
        model = nn.Linear(10, 100)
        tokenizer = type(
            "Tokenizer", (), {"encode": lambda s: [1, 2, 3], "pad_token_id": 0}
        )()

        lora_config = LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=[])
        loss_config = LossConfig(type="cross_entropy")

        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            lora_config=lora_config,
            loss_config=loss_config,
        )

        assert trainer.loss_config.type == "cross_entropy"

    def test_trainer_with_kl_divergence(self):
        """Test LoRATrainer with KL divergence loss."""
        model = nn.Linear(10, 100)
        tokenizer = type(
            "Tokenizer", (), {"encode": lambda s: [1, 2, 3], "pad_token_id": 0}
        )()

        lora_config = LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=[])
        loss_config = LossConfig(type="kl_divergence", temperature=2.0)

        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            lora_config=lora_config,
            loss_config=loss_config,
        )

        assert trainer.loss_config.type == "kl_divergence"
        assert trainer.loss_config.temperature == 2.0

    def test_trainer_with_mixed_loss(self):
        """Test LoRATrainer with mixed loss."""
        model = nn.Linear(10, 100)
        tokenizer = type(
            "Tokenizer", (), {"encode": lambda s: [1, 2, 3], "pad_token_id": 0}
        )()

        lora_config = LoRAConfig(rank=2, alpha=4, dropout=0.0, target_modules=[])
        loss_config = LossConfig(type="mixed", alpha=0.7, temperature=2.0)

        trainer = LoRATrainer(
            model=model,
            tokenizer=tokenizer,
            lora_config=lora_config,
            loss_config=loss_config,
        )

        assert trainer.loss_config.type == "mixed"
        assert trainer.loss_config.alpha == 0.7


class TestTokenLevelDisagreementWithLogits:
    """Test TokenLevelDisagreement with target_logits field."""

    def test_disagreement_with_logits(self):
        """Test creating disagreement with target logits."""
        disagreement = TokenLevelDisagreement(
            position=10,
            draft_token=100,
            target_token=200,
            draft_confidence=0.3,
            target_confidence=0.7,
            context_tokens=[1, 2, 3],
            target_logits=[(200, 0.7), (201, 0.2), (202, 0.1)],
        )

        assert disagreement.target_logits is not None
        assert len(disagreement.target_logits) == 3
        assert disagreement.target_logits[0] == (200, 0.7)

    def test_disagreement_without_logits(self):
        """Test creating disagreement without target logits (backward compat)."""
        disagreement = TokenLevelDisagreement(
            position=10,
            draft_token=100,
            target_token=200,
            draft_confidence=0.3,
            target_confidence=0.7,
            context_tokens=[1, 2, 3],
        )

        assert disagreement.target_logits is None

    def test_disagreement_serialization(self):
        """Test serializing and deserializing disagreement with logits."""
        disagreement = TokenLevelDisagreement(
            position=10,
            draft_token=100,
            target_token=200,
            draft_confidence=0.3,
            target_confidence=0.7,
            context_tokens=[1, 2, 3],
            target_logits=[(200, 0.7), (201, 0.2), (202, 0.1)],
        )

        # Serialize
        data = disagreement.to_dict()
        assert "target_logits" in data
        assert data["target_logits"] == [[200, 0.7], [201, 0.2], [202, 0.1]]

        # Deserialize
        restored = TokenLevelDisagreement.from_dict(data)
        assert restored.target_logits == [(200, 0.7), (201, 0.2), (202, 0.1)]

    def test_disagreement_backward_compat(self):
        """Test deserializing old format without target_logits."""
        data = {
            "position": 10,
            "draft_token": 100,
            "target_token": 200,
            "draft_confidence": 0.3,
            "target_confidence": 0.7,
            "context_tokens": [1, 2, 3],
        }

        disagreement = TokenLevelDisagreement.from_dict(data)
        assert disagreement.target_logits is None
