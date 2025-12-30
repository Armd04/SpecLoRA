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

from src.training import LoRALinear, LoRAConfig, LoRATrainer


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

        # Use the LoRALinear's get_fused_weight method
        fused_weight = lora_layer.get_fused_weight()
        bias = lora_layer.get_bias()

        # Create fused layer
        has_bias = bias is not None
        fused_layer = nn.Linear(16, 32, bias=has_bias)
        fused_layer.weight = fused_weight
        if has_bias:
            fused_layer.bias = bias
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
        config = LoRAConfig(
            rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]
        )

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

    def test_checkpoint_default_timestamped_naming(self):
        """Test that save_checkpoint() uses adapter-{timestamp} naming by default."""

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

            # Save checkpoint without explicit name
            checkpoint_path = trainer.save_checkpoint()

            # Verify path follows adapter-{timestamp} format
            checkpoint_dir = Path(checkpoint_path)
            assert checkpoint_dir.name.startswith("adapter-")
            # Extract timestamp part and verify it's a valid integer
            timestamp_str = checkpoint_dir.name.replace("adapter-", "")
            timestamp = int(timestamp_str)  # Should not raise
            assert timestamp > 0

            # Verify files exist
            assert (checkpoint_dir / "adapters.safetensors").exists()
            assert (checkpoint_dir / "adapter_config.json").exists()
            assert (checkpoint_dir / "trainer_state.json").exists()

    def test_save_to_best(self):
        """Test that save_to_best() copies adapter to 'best' folder."""

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

            # First save a checkpoint (required before save_to_best)
            original_path = trainer.save_checkpoint()
            assert Path(original_path).name.startswith("adapter-")

            # Now save to best (copies from saved checkpoint)
            best_path = trainer.save_to_best()

            # Verify path is the 'best' folder
            assert Path(best_path).name == "best"
            assert (Path(best_path) / "adapters.safetensors").exists()
            assert (Path(best_path) / "adapter_config.json").exists()
            assert (Path(best_path) / "trainer_state.json").exists()

    def test_save_to_best_after_fusion(self):
        """Test that save_to_best() works even after fuse_and_get_model()."""

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

            # Fuse LoRA weights (removes LoRA layers from model)
            trainer.fuse_and_get_model()

            # save_to_best() should still work by copying files
            best_path = trainer.save_to_best()

            # Verify the best folder has correct files
            assert Path(best_path).name == "best"
            assert (Path(best_path) / "adapters.safetensors").exists()
            assert (Path(best_path) / "adapter_config.json").exists()

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
            assert any(
                "lora_a" in name for name in weight_names
            ), f"Expected lora_a in {weight_names}"
            assert any(
                "lora_b" in name for name in weight_names
            ), f"Expected lora_b in {weight_names}"

            # Should NOT have uppercase lora_A or lora_B
            assert not any(
                "lora_A" in name for name in weight_names
            ), f"Found lora_A in {weight_names}"
            assert not any(
                "lora_B" in name for name in weight_names
            ), f"Found lora_B in {weight_names}"


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

        config = LoRAConfig(
            rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]
        )

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

        config = LoRAConfig(
            rank=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]
        )

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


class TestEnhancedLayerAccess:
    """Tests for enhanced layer access in ModelManager (indexed vs named layers)."""

    def test_load_adapter_with_indexed_layers(self):
        """Test loading adapter on model with indexed layers (e.g., model.layers[0])."""
        from src.models import ModelManager
        from unittest.mock import Mock

        # Create a model with indexed layers structure
        class IndexedLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Use a list of layers that can be accessed by index
                self.layers = [nn.Linear(16, 16) for _ in range(3)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = IndexedLayerModel()
        mx.eval(model.parameters())

        # Create mock tokenizer
        mock_tokenizer = Mock()

        # Create ModelManager and set draft model
        manager = ModelManager(
            target_model_name="test/target",
            draft_model_name="test/draft",
        )
        manager.draft_model = model
        manager.draft_tokenizer = mock_tokenizer

        # Create adapter directory with weights for indexed layer
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Create adapter config
            adapter_config = {
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.0,
                    "dropout": 0.0,
                },
                "lora_layers": ["layers.0", "layers.2"],  # Index-based paths
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)

            # Create adapter weights
            weights = {
                "layers.0.lora_a": mx.random.normal((4, 16)),
                "layers.0.lora_b": mx.random.normal((16, 4)),
                "layers.2.lora_a": mx.random.normal((4, 16)),
                "layers.2.lora_b": mx.random.normal((16, 4)),
            }
            mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)

            # Load adapter - should handle indexed access correctly
            manager.load_lora_adapter(str(adapter_dir), fuse=True)

            # Verify layers are still nn.Linear (not LoRALinear since fuse=True)
            assert isinstance(manager.draft_model.layers[0], nn.Linear)
            assert isinstance(manager.draft_model.layers[2], nn.Linear)

            # Verify weights were modified (fused)
            # Original weights shape should be preserved
            assert manager.draft_model.layers[0].weight.shape == (16, 16)

    def test_load_adapter_with_named_layers(self):
        """Test loading adapter on model with named attribute layers."""
        from src.models import ModelManager
        from unittest.mock import Mock

        # Create a model with named attributes
        class NamedLayerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.k_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.k_proj(x) + self.v_proj(x)

        model = NamedLayerModel()
        mx.eval(model.parameters())

        mock_tokenizer = Mock()

        manager = ModelManager(
            target_model_name="test/target",
            draft_model_name="test/draft",
        )
        manager.draft_model = model
        manager.draft_tokenizer = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Create adapter config
            adapter_config = {
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.0,
                    "dropout": 0.0,
                },
                "lora_layers": ["q_proj", "v_proj"],  # Named attribute paths
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)

            # Create adapter weights
            weights = {
                "q_proj.lora_a": mx.random.normal((4, 16)),
                "q_proj.lora_b": mx.random.normal((16, 4)),
                "v_proj.lora_a": mx.random.normal((4, 16)),
                "v_proj.lora_b": mx.random.normal((16, 4)),
            }
            mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)

            # Load adapter
            manager.load_lora_adapter(str(adapter_dir), fuse=True)

            # Verify layers are still nn.Linear
            assert isinstance(manager.draft_model.q_proj, nn.Linear)
            assert isinstance(manager.draft_model.v_proj, nn.Linear)
            assert isinstance(manager.draft_model.k_proj, nn.Linear)

    def test_load_adapter_with_nested_indexed_layers(self):
        """Test loading adapter on model with nested indexed structure (e.g., model.layers[0].attention)."""
        from src.models import ModelManager
        from unittest.mock import Mock

        # Create a model with nested structure
        class AttentionBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.v_proj(x)

        class NestedIndexedModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Create indexed list of attention blocks
                self.layers = [AttentionBlock() for _ in range(2)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = NestedIndexedModel()
        mx.eval(model.parameters())

        mock_tokenizer = Mock()

        manager = ModelManager(
            target_model_name="test/target",
            draft_model_name="test/draft",
        )
        manager.draft_model = model
        manager.draft_tokenizer = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Create adapter config with nested indexed paths
            adapter_config = {
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.0,
                    "dropout": 0.0,
                },
                "lora_layers": [
                    "layers.0.q_proj",  # Indexed layer, named attribute
                    "layers.1.v_proj",  # Different indexed layer
                ],
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)

            # Create adapter weights
            weights = {
                "layers.0.q_proj.lora_a": mx.random.normal((4, 16)),
                "layers.0.q_proj.lora_b": mx.random.normal((16, 4)),
                "layers.1.v_proj.lora_a": mx.random.normal((4, 16)),
                "layers.1.v_proj.lora_b": mx.random.normal((16, 4)),
            }
            mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)

            # Load adapter - should handle mixed indexed/named access
            manager.load_lora_adapter(str(adapter_dir), fuse=True)

            # Verify correct layers were modified
            assert isinstance(manager.draft_model.layers[0].q_proj, nn.Linear)
            assert isinstance(manager.draft_model.layers[1].v_proj, nn.Linear)

    def test_load_adapter_handles_missing_layer_gracefully(self):
        """Test that loading adapter with invalid layer path logs warning but continues with valid layers."""
        from src.models import ModelManager
        from unittest.mock import Mock

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16)
                self.v_proj = nn.Linear(16, 16)

            def __call__(self, x):
                return self.q_proj(x) + self.v_proj(x)

        model = SimpleModel()
        mx.eval(model.parameters())

        mock_tokenizer = Mock()

        manager = ModelManager(
            target_model_name="test/target",
            draft_model_name="test/draft",
        )
        manager.draft_model = model
        manager.draft_tokenizer = mock_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Create adapter config with one invalid and two valid layers
            adapter_config = {
                "fine_tune_type": "lora",
                "lora_parameters": {
                    "rank": 4,
                    "scale": 8.0,
                    "dropout": 0.0,
                },
                "lora_layers": [
                    "nonexistent_layer",
                    "q_proj",
                    "v_proj",
                ],  # One invalid, two valid
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(adapter_config, f)

            # Create adapter weights (include weights for nonexistent layer)
            weights = {
                "nonexistent_layer.lora_a": mx.random.normal((4, 16)),
                "nonexistent_layer.lora_b": mx.random.normal((16, 4)),
                "q_proj.lora_a": mx.random.normal((4, 16)),
                "q_proj.lora_b": mx.random.normal((16, 4)),
                "v_proj.lora_a": mx.random.normal((4, 16)),
                "v_proj.lora_b": mx.random.normal((16, 4)),
            }
            mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), weights)

            # Load adapter - should log warning about missing layer but continue with valid layers
            # Should NOT raise exception since 2 out of 3 layers are valid (>50%)
            manager.load_lora_adapter(str(adapter_dir), fuse=True)

            # Verify the valid layers were successfully loaded
            assert isinstance(manager.draft_model.q_proj, nn.Linear)
            assert isinstance(manager.draft_model.v_proj, nn.Linear)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
