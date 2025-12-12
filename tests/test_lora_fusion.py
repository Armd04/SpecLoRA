"""
Tests for LoRA weight fusion functionality.

These tests verify that:
1. LoRA weights are correctly fused into base model weights
2. Fused models produce the same output as wrapped models
3. Checkpoint saving produces MLX-LM compatible format
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx.nn as nn

from src.training import LoRALinear, LoRAConfig, LoRATrainer, apply_lora_to_model


class TestLoRALinear:
    """Tests for LoRALinear adapter layer."""

    def test_lora_linear_forward(self):
        """Test that LoRALinear forward pass works correctly."""
        # Create a simple linear layer
        original = nn.Linear(16, 32)
        mx.eval(original.parameters())

        # Wrap with LoRA
        lora_layer = LoRALinear(original, rank=4, alpha=8)
        mx.eval(lora_layer.parameters())

        # Forward pass
        x = mx.random.normal((2, 16))
        output = lora_layer(x)

        assert output.shape == (2, 32)

    def test_lora_linear_initial_identity(self):
        """Test that LoRA starts as identity (no change to output)."""
        # Create a simple linear layer
        original = nn.Linear(16, 32)
        mx.eval(original.parameters())

        # Wrap with LoRA - B is initialized to zeros, so output should be same as original
        lora_layer = LoRALinear(original, rank=4, alpha=8)
        lora_layer._training = False  # Disable dropout
        mx.eval(lora_layer.parameters())

        # Forward pass
        x = mx.random.normal((2, 16))
        original_output = original(x)
        lora_output = lora_layer(x)

        # Since lora_B is initialized to zeros, outputs should be nearly identical
        # (small difference due to lora_A random init, but scaled by zero B)
        mx.eval(original_output)
        mx.eval(lora_output)
        assert mx.allclose(original_output, lora_output, atol=1e-5).item()


class TestLoRAFusion:
    """Tests for LoRA weight fusion into base model."""

    def test_fuse_lora_linear(self):
        """Test fusing a single LoRALinear layer."""
        # Create original layer
        original = nn.Linear(16, 32)
        mx.eval(original.parameters())

        # Save original weight before wrapping
        original_weight = original.weight

        # Wrap with LoRA and set non-zero B weights
        lora_layer = LoRALinear(original, rank=4, alpha=8)
        lora_layer.lora_B = mx.random.normal((32, 4)) * 0.1
        lora_layer._training = False
        mx.eval(lora_layer.parameters())

        # Create test input
        x = mx.random.normal((2, 16))

        # Get output from LoRA layer
        lora_output = lora_layer(x)
        mx.eval(lora_output)

        # Manually fuse weights - use the weight from original_layer inside LoRALinear
        scaling = lora_layer.alpha / lora_layer.rank
        delta = (lora_layer.lora_B @ lora_layer.lora_A) * scaling
        fused_weight = lora_layer.original_layer.weight + delta

        # Create fused layer (need to handle bias properly)
        has_bias = hasattr(lora_layer.original_layer, 'bias') and lora_layer.original_layer.bias is not None
        fused_layer = nn.Linear(16, 32, bias=has_bias)
        fused_layer.weight = fused_weight
        if has_bias:
            fused_layer.bias = lora_layer.original_layer.bias
        mx.eval(fused_layer.parameters())

        # Get output from fused layer
        fused_output = fused_layer(x)
        mx.eval(fused_output)

        # Outputs should be identical
        assert mx.allclose(lora_output, fused_output, atol=1e-5).item()


class TestCheckpointFormat:
    """Tests for MLX-LM compatible checkpoint format."""

    def test_checkpoint_has_adapter_config(self):
        """Test that saved checkpoints include adapter_config.json."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.v_proj(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [1, 2, 3]

        tokenizer = MockTokenizer()

        # Create LoRA config and trainer
        config = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoRATrainer(
                model=model,
                tokenizer=tokenizer,
                lora_config=config,
                checkpoint_dir=tmpdir,
            )

            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint("test")

            # Verify files exist
            checkpoint_dir = Path(checkpoint_path)
            assert (checkpoint_dir / "adapters.safetensors").exists()
            assert (checkpoint_dir / "adapter_config.json").exists()
            assert (checkpoint_dir / "trainer_state.json").exists()

            # Verify adapter_config.json format
            with open(checkpoint_dir / "adapter_config.json") as f:
                adapter_config = json.load(f)

            assert adapter_config["fine_tune_type"] == "lora"
            assert "lora_parameters" in adapter_config
            assert adapter_config["lora_parameters"]["rank"] == 4
            assert adapter_config["lora_parameters"]["scale"] == 8.0

    def test_checkpoint_uses_mlx_lm_naming(self):
        """Test that saved weights use MLX-LM naming convention (lora_a, lora_b)."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [1, 2, 3]

        tokenizer = MockTokenizer()

        config = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoRATrainer(
                model=model,
                tokenizer=tokenizer,
                lora_config=config,
                checkpoint_dir=tmpdir,
            )

            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint("test")

            # Load weights and check naming
            weights = mx.load(str(Path(checkpoint_path) / "adapters.safetensors"))

            # Should have lowercase lora_a and lora_b (MLX-LM format)
            weight_names = list(weights.keys())
            assert any("lora_a" in name for name in weight_names), f"Expected lora_a in {weight_names}"
            assert any("lora_b" in name for name in weight_names), f"Expected lora_b in {weight_names}"

            # Should NOT have uppercase lora_A or lora_B
            assert not any("lora_A" in name for name in weight_names), f"Found lora_A in {weight_names}"
            assert not any("lora_B" in name for name in weight_names), f"Found lora_B in {weight_names}"


class TestTrainerFusion:
    """Tests for LoRATrainer.fuse_and_get_model() method."""

    def test_fuse_and_get_model_returns_clean_model(self):
        """Test that fuse_and_get_model returns a model without LoRA wrappers."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.v_proj(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [1, 2, 3]

        tokenizer = MockTokenizer()

        config = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoRATrainer(
                model=model,
                tokenizer=tokenizer,
                lora_config=config,
                checkpoint_dir=tmpdir,
            )

            # Before fusion, model should have LoRALinear layers
            assert isinstance(trainer.model.q_proj, LoRALinear)
            assert isinstance(trainer.model.v_proj, LoRALinear)

            # Fuse and get clean model
            clean_model = trainer.fuse_and_get_model()

            # After fusion, model should have regular Linear layers
            assert isinstance(clean_model.q_proj, nn.Linear)
            assert isinstance(clean_model.v_proj, nn.Linear)
            assert not isinstance(clean_model.q_proj, LoRALinear)
            assert not isinstance(clean_model.v_proj, LoRALinear)

    def test_fused_model_produces_same_output(self):
        """Test that fused model produces same output as LoRA-wrapped model."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.v_proj(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            def encode(self, text):
                return [1, 2, 3]

        tokenizer = MockTokenizer()

        config = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = LoRATrainer(
                model=model,
                tokenizer=tokenizer,
                lora_config=config,
                checkpoint_dir=tmpdir,
            )

            # Set non-zero LoRA weights for testing
            trainer.model.q_proj.lora_B = mx.random.normal((16, 4)) * 0.1
            trainer.model.v_proj.lora_B = mx.random.normal((16, 4)) * 0.1
            trainer.model.q_proj._training = False
            trainer.model.v_proj._training = False
            mx.eval(trainer.model.parameters())

            # Get output from LoRA model
            x = mx.random.normal((2, 16))
            lora_output = trainer.model(x)
            mx.eval(lora_output)

            # Fuse and get clean model
            clean_model = trainer.fuse_and_get_model()

            # Get output from fused model
            fused_output = clean_model(x)
            mx.eval(fused_output)

            # Outputs should be identical
            assert mx.allclose(lora_output, fused_output, atol=1e-5).item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
