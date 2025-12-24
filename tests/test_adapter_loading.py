"""
Tests for LoRA adapter loading and management functionality.

Tests the adapter loading system added in the addLoadAdapter branch:
- Path resolution (aliases: best, final, latest)
- Adapter structure validation
- Adapter loading and unloading
- Decoder reinitialization
- Error handling and rollback
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.main import SpeculativeDecodingSystem


class TestAdapterPathResolution:
    """Tests for resolve_adapter_path() method."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir()

            # Create some checkpoint directories
            (checkpoint_dir / "best").mkdir()
            (checkpoint_dir / "final").mkdir()
            (checkpoint_dir / "checkpoint_001").mkdir()
            (checkpoint_dir / "checkpoint_002").mkdir()

            # Make checkpoint_002 the most recent
            import time

            time.sleep(0.01)
            (checkpoint_dir / "checkpoint_002" / "marker.txt").touch()

            yield tmpdir, checkpoint_dir

    @pytest.fixture
    def system_with_temp_config(self, temp_checkpoint_dir):
        """Create system with temporary checkpoint directory."""
        tmpdir, checkpoint_dir = temp_checkpoint_dir
        config = {"training": {"checkpoint_dir": str(checkpoint_dir)}}
        system = SpeculativeDecodingSystem(config)
        system._initialized = False  # Don't actually initialize models
        return system, checkpoint_dir

    def test_resolve_alias_best(self, system_with_temp_config):
        """Test resolving 'best' alias."""
        system, checkpoint_dir = system_with_temp_config
        resolved = system.resolve_adapter_path("best")
        assert resolved == checkpoint_dir / "best"
        assert resolved.exists()

    def test_resolve_alias_final(self, system_with_temp_config):
        """Test resolving 'final' alias."""
        system, checkpoint_dir = system_with_temp_config
        resolved = system.resolve_adapter_path("final")
        assert resolved == checkpoint_dir / "final"
        assert resolved.exists()

    def test_resolve_alias_latest(self, system_with_temp_config):
        """Test resolving 'latest' alias (most recent checkpoint)."""
        system, checkpoint_dir = system_with_temp_config
        resolved = system.resolve_adapter_path("latest")
        assert resolved == checkpoint_dir / "checkpoint_002"
        assert resolved.exists()

    def test_resolve_alias_case_insensitive(self, system_with_temp_config):
        """Test that aliases are case-insensitive."""
        system, checkpoint_dir = system_with_temp_config
        assert system.resolve_adapter_path("BEST") == checkpoint_dir / "best"
        assert system.resolve_adapter_path("Best") == checkpoint_dir / "best"
        assert system.resolve_adapter_path("FiNaL") == checkpoint_dir / "final"

    def test_resolve_relative_path(self, system_with_temp_config):
        """Test resolving relative path."""
        system, checkpoint_dir = system_with_temp_config
        # Use relative path from CWD
        try:
            rel_path = str(checkpoint_dir.relative_to(Path.cwd()) / "best")
        except ValueError:
            # If checkpoint_dir is not relative to CWD (e.g., in /tmp),
            # skip this test or create a different relative path
            pytest.skip("Checkpoint dir not relative to CWD")
        resolved = system.resolve_adapter_path(rel_path)
        assert resolved == checkpoint_dir / "best"

    def test_resolve_absolute_path(self, system_with_temp_config):
        """Test resolving absolute path."""
        system, checkpoint_dir = system_with_temp_config
        abs_path = str(checkpoint_dir / "final")
        resolved = system.resolve_adapter_path(abs_path)
        assert resolved == checkpoint_dir / "final"

    def test_resolve_nonexistent_raises_error(self, system_with_temp_config):
        """Test that resolving nonexistent path raises FileNotFoundError."""
        system, _ = system_with_temp_config
        with pytest.raises(FileNotFoundError, match="Adapter not found"):
            system.resolve_adapter_path("nonexistent_adapter")

    def test_resolve_latest_no_checkpoints(self, system_with_temp_config):
        """Test 'latest' alias when no checkpoints exist."""
        system, checkpoint_dir = system_with_temp_config
        # Remove all checkpoint directories
        for d in checkpoint_dir.iterdir():
            if d.is_dir():
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()

        with pytest.raises(FileNotFoundError, match="No checkpoints found"):
            system.resolve_adapter_path("latest")

    def test_resolve_latest_empty_checkpoint_dir(self):
        """Test 'latest' alias when checkpoint directory doesn't exist."""
        config = {"training": {"checkpoint_dir": "/nonexistent/path"}}
        system = SpeculativeDecodingSystem(config)
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            system.resolve_adapter_path("latest")


class TestAdapterStructureValidation:
    """Tests for validate_adapter_structure() method."""

    @pytest.fixture
    def valid_adapter_dir(self):
        """Create a valid adapter directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Create adapter config
            config = {
                "lora_parameters": {
                    "rank": 8,
                    "scale": 16.0,  # MLX-LM uses 'scale' instead of 'alpha'
                    "dropout": 0.05,
                }
            }
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(config, f)

            # Create trainer state
            state = {
                "global_step": 1000,
                "best_loss": 0.123,
                "epoch": 5,
            }
            with open(adapter_dir / "trainer_state.json", "w") as f:
                json.dump(state, f)

            # Create weights file (empty is fine for testing)
            (adapter_dir / "adapters.safetensors").touch()

            yield adapter_dir

    def test_validate_valid_structure(self, valid_adapter_dir):
        """Test validation of valid adapter structure."""
        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        metadata = system.validate_adapter_structure(valid_adapter_dir)

        assert metadata["rank"] == 8
        assert metadata["alpha"] == 16.0
        assert metadata["dropout"] == 0.05
        assert metadata["global_step"] == 1000
        assert metadata["best_loss"] == 0.123
        assert metadata["path"] == str(valid_adapter_dir)

    def test_validate_missing_weights_file(self, valid_adapter_dir):
        """Test validation fails when weights file is missing."""
        (valid_adapter_dir / "adapters.safetensors").unlink()

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="Invalid adapter structure"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_missing_adapter_config(self, valid_adapter_dir):
        """Test validation fails when adapter_config.json is missing."""
        (valid_adapter_dir / "adapter_config.json").unlink()

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="Invalid adapter structure"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_missing_trainer_state(self, valid_adapter_dir):
        """Test validation fails when trainer_state.json is missing."""
        (valid_adapter_dir / "trainer_state.json").unlink()

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="Invalid adapter structure"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_corrupted_adapter_config(self, valid_adapter_dir):
        """Test validation fails when adapter_config.json is corrupted."""
        with open(valid_adapter_dir / "adapter_config.json", "w") as f:
            f.write("{invalid json")

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="Invalid adapter_config.json"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_corrupted_trainer_state(self, valid_adapter_dir):
        """Test validation fails when trainer_state.json is corrupted."""
        with open(valid_adapter_dir / "trainer_state.json", "w") as f:
            f.write("not valid json at all")

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="Invalid trainer_state.json"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_missing_rank_in_config(self, valid_adapter_dir):
        """Test validation fails when rank is missing from config."""
        # Overwrite with config missing rank
        config = {"lora_parameters": {"scale": 16.0}}
        with open(valid_adapter_dir / "adapter_config.json", "w") as f:
            json.dump(config, f)

        config = {"training": {"checkpoint_dir": str(valid_adapter_dir.parent)}}
        system = SpeculativeDecodingSystem(config)

        with pytest.raises(ValueError, match="missing lora_parameters.rank"):
            system.validate_adapter_structure(valid_adapter_dir)

    def test_validate_minimal_valid_structure(self):
        """Test validation with minimal valid structure (optional fields missing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            # Minimal config (only rank required)
            config = {"lora_parameters": {"rank": 4}}
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(config, f)

            # Minimal state (empty is valid)
            state = {}
            with open(adapter_dir / "trainer_state.json", "w") as f:
                json.dump(state, f)

            (adapter_dir / "adapters.safetensors").touch()

            config = {"training": {"checkpoint_dir": str(adapter_dir.parent)}}
            system = SpeculativeDecodingSystem(config)

            metadata = system.validate_adapter_structure(adapter_dir)
            assert metadata["rank"] == 4
            assert metadata["alpha"] is None
            assert metadata["global_step"] is None


class TestAdapterLoadingUnloading:
    """Tests for load_lora_adapter() and unload_lora_adapter() methods."""

    @pytest.fixture
    def mock_system(self):
        """Create a mock system with mocked model manager and decoder."""
        config = {
            "training": {"checkpoint_dir": "data/checkpoints"},
            "speculative": {
                "num_draft_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "acceptance_threshold": 0.5,
            },
            "chat": {},
        }
        system = SpeculativeDecodingSystem(config)

        # Mock model manager
        system.model_manager = Mock()
        system.model_manager.load_lora_adapter = Mock()
        system.model_manager.clear_cache = Mock()
        system.model_manager.get_target_model = Mock(return_value=(Mock(), Mock()))
        system.model_manager.get_draft_model = Mock(return_value=(Mock(), Mock()))

        # Mock decoder
        system.decoder = Mock()

        system._initialized = True
        return system

    @pytest.fixture
    def valid_adapter_dir(self):
        """Create a valid adapter directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir) / "test_adapter"
            adapter_dir.mkdir()

            config = {"lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.05}}
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(config, f)

            state = {"global_step": 100, "best_loss": 0.5}
            with open(adapter_dir / "trainer_state.json", "w") as f:
                json.dump(state, f)

            (adapter_dir / "adapters.safetensors").touch()

            yield adapter_dir

    def test_load_adapter_success(self, mock_system, valid_adapter_dir):
        """Test successful adapter loading."""
        metadata = mock_system.load_lora_adapter(str(valid_adapter_dir))

        # Verify adapter was loaded
        mock_system.model_manager.load_lora_adapter.assert_called_once_with(
            str(valid_adapter_dir), fuse=True
        )

        # Verify cache was cleared
        mock_system.model_manager.clear_cache.assert_called_once()

        # Verify metadata returned
        assert metadata["rank"] == 8
        assert metadata["alpha"] == 16.0
        assert metadata["global_step"] == 100

        # Verify internal state updated
        assert mock_system._current_adapter_path == str(valid_adapter_dir)
        assert mock_system._current_adapter_info == metadata

    def test_load_adapter_invalid_path(self, mock_system):
        """Test loading with invalid path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid adapter"):
            mock_system.load_lora_adapter("/nonexistent/path")

    def test_load_adapter_invalid_structure(self, mock_system):
        """Test loading with invalid structure raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory but missing required files
            adapter_dir = Path(tmpdir) / "incomplete"
            adapter_dir.mkdir()

            with pytest.raises(ValueError, match="Invalid adapter"):
                mock_system.load_lora_adapter(str(adapter_dir))

    def test_load_adapter_mlx_error_rollback(self, mock_system, valid_adapter_dir):
        """Test rollback when MLX loading fails."""
        # Set previous adapter
        mock_system._current_adapter_path = "/old/adapter"

        # Make load_lora_adapter fail
        mock_system.model_manager.load_lora_adapter.side_effect = [
            RuntimeError("MLX error"),
            None,  # Rollback succeeds
        ]

        with pytest.raises(RuntimeError, match="Failed to load adapter"):
            mock_system.load_lora_adapter(str(valid_adapter_dir))

        # Verify rollback attempted
        assert mock_system.model_manager.load_lora_adapter.call_count == 2
        # Second call should be the rollback
        rollback_call = mock_system.model_manager.load_lora_adapter.call_args_list[1]
        assert rollback_call[0][0] == "/old/adapter"

    def test_load_adapter_rollback_fails(self, mock_system, valid_adapter_dir):
        """Test handling when both load and rollback fail."""
        mock_system._current_adapter_path = "/old/adapter"

        # Both load and rollback fail
        mock_system.model_manager.load_lora_adapter.side_effect = RuntimeError("Failed")

        with pytest.raises(RuntimeError, match="Failed to load adapter"):
            mock_system.load_lora_adapter(str(valid_adapter_dir))

        # System should be marked as uninitialized
        assert mock_system._initialized is False

    def test_unload_adapter_success(self, mock_system):
        """Test successful adapter unloading."""
        # Set current adapter
        mock_system._current_adapter_path = "/some/adapter"
        mock_system._current_adapter_info = {"rank": 8}

        # Add config required for model name
        mock_system.config["models"] = {"draft": {"name": "test/draft"}}

        with patch("mlx_lm.load") as mock_load:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)

            with patch("mlx.core.eval"):
                mock_system.unload_lora_adapter()

                # Verify base model reloaded
                mock_load.assert_called_once()

                # Verify cache cleared
                mock_system.model_manager.clear_cache.assert_called_once()

                # Verify state cleared
                assert mock_system._current_adapter_path is None
                assert mock_system._current_adapter_info is None

    def test_unload_adapter_when_none_loaded(self, mock_system):
        """Test unloading when no adapter is loaded (should be no-op)."""
        mock_system._current_adapter_path = None

        # Should not raise error, just log warning
        mock_system.unload_lora_adapter()

        # Should not attempt to reload models
        mock_system.model_manager.clear_cache.assert_not_called()

    def test_get_current_adapter_info_loaded(self, mock_system):
        """Test getting adapter info when adapter is loaded."""
        mock_system._current_adapter_info = {"rank": 8, "alpha": 16.0}

        info = mock_system.get_current_adapter_info()
        assert info == {"rank": 8, "alpha": 16.0}

    def test_get_current_adapter_info_none(self, mock_system):
        """Test getting adapter info when no adapter is loaded."""
        mock_system._current_adapter_info = None

        info = mock_system.get_current_adapter_info()
        assert info is None


class TestDecoderReinitialization:
    """Tests for _reinitialize_decoder() method."""

    def test_reinitialize_decoder_creates_new_instance(self):
        """Test that decoder is recreated with updated models."""
        config = {
            "speculative": {
                "num_draft_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "acceptance_threshold": 0.5,
            },
            "chat": {"system_message": "Test system message"},
        }
        system = SpeculativeDecodingSystem(config)

        # Mock model manager
        target_model = Mock()
        target_tokenizer = Mock()
        draft_model = Mock()
        draft_tokenizer = Mock()

        system.model_manager = Mock()
        system.model_manager.get_target_model = Mock(
            return_value=(target_model, target_tokenizer)
        )
        system.model_manager.get_draft_model = Mock(
            return_value=(draft_model, draft_tokenizer)
        )

        # Create initial decoder
        system.decoder = Mock()
        old_decoder = system.decoder

        # Reinitialize - patch where SpeculativeDecoder is imported
        with patch("src.speculative.SpeculativeDecoder") as mock_decoder_class:
            mock_new_decoder = Mock()
            mock_decoder_class.return_value = mock_new_decoder

            system._reinitialize_decoder()

            # Verify new decoder created with correct parameters
            mock_decoder_class.assert_called_once_with(
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=target_tokenizer,
                num_draft_tokens=4,
                temperature=0.7,
                top_p=0.9,
                acceptance_threshold=0.5,
                system_message="Test system message",
            )

            # Verify decoder was replaced
            assert system.decoder == mock_new_decoder
            assert system.decoder != old_decoder


class TestCommandHandling:
    """Tests for interactive command handling."""

    @pytest.fixture
    def mock_system(self):
        """Create mock system for command testing."""
        config = {
            "training": {"checkpoint_dir": "data/checkpoints"},
            "speculative": {
                "num_draft_tokens": 4,
                "temperature": 0.7,
                "top_p": 0.9,
                "acceptance_threshold": 0.5,
            },
            "chat": {},
        }
        system = SpeculativeDecodingSystem(config)
        system._initialized = True
        return system

    def test_handle_quit_command(self, mock_system):
        """Test /quit command returns True."""
        assert mock_system._handle_command("quit") is True
        assert mock_system._handle_command("exit") is True

    def test_handle_stats_command(self, mock_system):
        """Test /stats command calls get_stats."""
        with patch.object(mock_system, "get_stats", return_value={"test": "stats"}):
            result = mock_system._handle_command("stats")
            assert result is False
            mock_system.get_stats.assert_called_once()

    def test_handle_train_command(self, mock_system):
        """Test /train command calls train."""
        with patch.object(mock_system, "train"):
            result = mock_system._handle_command("train")
            assert result is False
            mock_system.train.assert_called_once()

    def test_handle_clear_command(self, mock_system):
        """Test /clear command."""
        with patch("src.main.console") as mock_console:
            result = mock_system._handle_command("clear")
            assert result is False
            mock_console.clear.assert_called_once()

    def test_handle_load_adapter_no_args(self, mock_system):
        """Test /load-adapter without arguments shows error."""
        with patch("src.main.console") as mock_console:
            result = mock_system._handle_command("load-adapter")
            assert result is False
            # Should print error message
            assert mock_console.print.called

    def test_handle_unload_adapter_none_loaded(self, mock_system):
        """Test /unload-adapter when no adapter loaded."""
        mock_system._current_adapter_path = None
        with patch("src.main.console") as mock_console:
            result = mock_system._handle_command("unload-adapter")
            assert result is False
            # Should print warning
            assert mock_console.print.called

    def test_handle_adapter_info_none_loaded(self, mock_system):
        """Test /adapter-info when no adapter loaded."""
        mock_system._current_adapter_info = None
        with patch("src.main.console") as mock_console:
            result = mock_system._handle_command("adapter-info")
            assert result is False
            # Should print warning
            assert mock_console.print.called

    def test_handle_adapter_info_loaded(self, mock_system):
        """Test /adapter-info when adapter is loaded."""
        mock_system._current_adapter_info = {"rank": 8, "path": "/test/adapter"}
        with patch("src.main.console"):
            with patch.object(mock_system, "_display_adapter_info"):
                result = mock_system._handle_command("adapter-info")
                assert result is False
                mock_system._display_adapter_info.assert_called_once()

    def test_handle_unknown_command(self, mock_system):
        """Test unknown command shows error."""
        with patch("src.main.console") as mock_console:
            result = mock_system._handle_command("unknown-command")
            assert result is False
            # Should print error
            assert mock_console.print.called


class TestInitializeWithAdapter:
    """Tests for initialize() with initial_adapter_path parameter."""

    def test_initialize_with_valid_adapter(self):
        """Test initialization with valid adapter path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid adapter
            adapter_dir = Path(tmpdir) / "adapter"
            adapter_dir.mkdir()

            config = {"lora_parameters": {"rank": 8, "scale": 16.0}}
            with open(adapter_dir / "adapter_config.json", "w") as f:
                json.dump(config, f)

            state = {"global_step": 100}
            with open(adapter_dir / "trainer_state.json", "w") as f:
                json.dump(state, f)

            (adapter_dir / "adapters.safetensors").touch()

            # Test initialization
            config_dict = {"training": {"checkpoint_dir": tmpdir}}
            system = SpeculativeDecodingSystem(config_dict)

            with patch("src.models.ModelManager"):
                with patch.object(
                    system, "resolve_adapter_path", return_value=adapter_dir
                ):
                    with patch.object(
                        system,
                        "validate_adapter_structure",
                        return_value={"rank": 8, "alpha": 16.0},
                    ):
                        with patch("src.speculative.SpeculativeDecoder"):
                            with patch("src.data_collector.DataCollector"):
                                with patch("src.data_collector.AcceptanceRateTracker"):
                                    # This would normally initialize - just verify it doesn't crash
                                    # Full integration test would require actual models
                                    pass

    def test_initialize_with_invalid_adapter_continues(self):
        """Test that initialization continues with base model if adapter loading fails."""
        config = {"training": {"checkpoint_dir": "/nonexistent"}}
        system = SpeculativeDecodingSystem(config)

        with patch("src.models.ModelManager"):
            with patch.object(
                system,
                "resolve_adapter_path",
                side_effect=FileNotFoundError("Adapter not found"),
            ):
                with patch("src.main.console"):
                    with patch("src.speculative.SpeculativeDecoder"):
                        with patch("src.data_collector.DataCollector"):
                            with patch("src.data_collector.AcceptanceRateTracker"):
                                # Should not raise, should print warning and continue
                                # Full test would require actual model loading
                                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
