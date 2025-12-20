# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpecLoRA is a speculative decoding system with adaptive LoRA training for Apple Silicon. It combines two inference optimization techniques:

1. **Speculative Decoding**: Uses a small draft model (Qwen2.5-0.5B) to generate K candidate tokens quickly, then verifies them in parallel with a larger target model (Qwen2.5-3B/7B). Accepted tokens are kept; rejected tokens are resampled.

2. **Adaptive LoRA Training**: When draft model acceptance rates drop below threshold, the system collects "failure cases" and fine-tunes the draft model using Low-Rank Adaptation (LoRA) to better match the target model's behavior.

This creates a feedback loop where the draft model continuously improves, increasing acceptance rates and overall inference speed over time.

## Key Commands

### Development Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify MLX installation (must show GPU device)
python -c "import mlx.core as mx; print(mx.default_device())"
```

### Running the System

**Important**: Always activate the virtual environment before running any commands:
```bash
source venv/bin/activate
```

Then run the system:

```bash
# Generate text with speculative decoding (production mode - fast)
python -m src.main generate "Your prompt here" --mode fast

# Generate with token-level data collection (detailed mode - for training)
python -m src.main generate "Your prompt here" --mode detailed

# Batch data collection for training
python -m src.main collect-data --prompts-file prompts.txt

# Train the draft model on collected failure cases
python -m src.main train

# Interactive mode (includes /stats, /train, /eval commands)
python -m src.main interactive

# Run demo
python -m src.main demo
```

### Testing

**Note**: Ensure venv is activated (`source venv/bin/activate`) before running tests.

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_lora_fusion.py

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_kv_cache.py::test_cache_rewind
```

### Code Quality

```bash
# Run ruff linting and formatting (used in pre-commit hooks)
ruff check . --fix
ruff format .

# Install pre-commit hooks
pre-commit install

# Run pre-commit manually
pre-commit run --all-files
```

## Architecture Overview

### Core Components

The system is organized into distinct modules with clear responsibilities:

- **`src/models.py`**: Model loading and management
  - `ModelManager`: Loads target and draft models with MLX
  - `load_lora_adapter()`: Loads LoRA weights and either fuses them into base model (inference) or wraps layers with LoRALinear (training)
  - KV cache utilities: `create_kv_cache()`, `get_logits_with_cache()`, `rewind_cache()`
  - Supports both regular `nn.Linear` and `nn.QuantizedLinear` layers

- **`src/speculative.py`**: Fast speculative decoding using MLX-LM
  - `SpeculativeDecoder`: Wraps MLX-LM's built-in speculative decoding
  - Production mode for inference (fast, minimal overhead)
  - Tracks acceptance rates but not token-level disagreements
  - Returns `SpeculativeResult` with text, tokens, and metrics

- **`src/speculative_manual.py`**: Manual speculative decoding for data collection
  - `ManualSpeculativeDecoder`: Custom implementation with token-level tracking
  - Captures every draft/target disagreement with confidence scores
  - Comparable speed to built-in version via aggressive KV caching
  - Returns `ManualSpeculativeResult` with detailed `TokenLevelDisagreement` records
  - Used when `--mode detailed` is specified

- **`src/data_collector.py`**: Failure case collection and replay buffer
  - `TrainingExample`: Stores prompt, target output, acceptance rate, and disagreements
  - `TokenLevelDisagreement`: Captures position, draft/target tokens, and confidence scores
  - `DataCollector`: Saves failures to JSONL in `data/failures/`
  - `AcceptanceRateTracker`: Tracks rolling acceptance rate statistics
  - Maintains replay buffer of successful cases to prevent catastrophic forgetting

- **`src/training.py`**: LoRA training pipeline
  - `LoRALinear`: LoRA adapter layer supporting both regular and quantized base layers
  - `LoRATrainer`: Fine-tunes draft model on collected failure cases
  - Mixed training: combines failure cases (80%) with replay buffer (20%)
  - Gradient checkpointing for memory efficiency
  - Saves checkpoints to `data/checkpoints/`
  - **Important**: Handles quantized layers by dequantizing during training, then re-quantizing during fusion

- **`src/main.py`**: CLI orchestration and main entry point
  - `SpeculativeDecodingSystem`: Coordinates all components
  - CLI commands: `generate`, `train`, `interactive`, `demo`, `collect-data`
  - Automatic cache clearing every N generations (configurable)
  - Generation count tracking for triggering training

### Two-Mode Architecture

The system operates in two distinct modes:

1. **Fast Mode** (`--mode fast`, default):
   - Uses `SpeculativeDecoder` (wraps MLX-LM built-in)
   - Optimized for production inference
   - Tracks overall acceptance rates only
   - No token-level data collection overhead

2. **Detailed Mode** (`--mode detailed`):
   - Uses `ManualSpeculativeDecoder` (custom implementation)
   - Captures token-level disagreements for training
   - Comparable speed via efficient KV cache management
   - Essential for collecting training data

### LoRA Fusion and Training Flow

Understanding the LoRA lifecycle is critical:

1. **Training Phase**:
   - Draft model loads in FP16 or quantized format
   - If quantized, weights are dequantized before wrapping with `LoRALinear`
   - Only LoRA parameters (A, B matrices) are trainable (~0.5% of params)
   - Checkpoints save both LoRA weights and adapter config

2. **Inference Phase**:
   - Load LoRA adapter with `fuse=True` (default in `load_lora_adapter()`)
   - LoRA weights are directly added to base weights: `W' = W + BA`
   - If base was quantized, the fused weights can be re-quantized
   - Result: single-layer inference with no overhead

3. **Continued Training**:
   - Load LoRA adapter with `fuse=False`
   - Layers are wrapped with `LoRALinear` to keep A, B separate
   - Enables gradient flow through LoRA parameters only

### KV Cache Management

Both decoders use KV caching for efficiency:

- **Draft model cache**: Speculatively extended during drafting, rewound on rejections
- **Target model cache**: Extended during verification, rewound, then updated with only accepted tokens
- **Rewinding**: Critical for maintaining cache validity when draft tokens are rejected
- **Performance**: Reduces attention computation from O(N+K) to O(K) per verification step

See `src/models.py` for cache utilities: `create_kv_cache()`, `get_logits_with_cache()`, `rewind_cache()`, `get_cache_length()`.

## Configuration

All settings in `configs/config.yaml`:

### Key Settings to Understand

- **`models.target.name`**: Larger model for verification (Qwen2.5-3B or 7B, quantized)
- **`models.draft.name`**: Smaller model for drafting (Qwen2.5-0.5B, quantized)
- **`speculative.num_draft_tokens`**: K tokens to draft per iteration (typically 4)
- **`speculative.acceptance_threshold`**: Below this rate, case is marked as failure (0.5 = 50%)
- **`training.lora.rank`**: LoRA rank (lower = less memory, less capacity)
- **`training.min_failure_cases`**: Minimum failures before triggering training
- **`training.replay_ratio`**: Fraction of training batch from replay buffer (prevents forgetting)
- **`memory.gradient_checkpointing`**: Enable for reduced memory during training
- **`memory.cache_clear_frequency`**: Clear MLX cache every N generations

### Memory Considerations

- **16GB RAM**: Use Qwen2.5-3B-Instruct-4bit target, rank=8, batch_size=1
- **32GB+ RAM**: Can use Qwen2.5-7B-Instruct-4bit target, higher rank, larger batch
- **Gradient accumulation**: Simulates larger batch sizes without memory overhead
- **Cache clearing**: Happens automatically in `src/main.py` based on generation count

## Data Flow

### Inference → Collection → Training Loop

```
1. User prompt → format_prompt() with tokenizer.apply_chat_template()
2. SpeculativeDecoder.generate() (fast) or ManualSpeculativeDecoder.generate() (detailed)
3. If acceptance_rate < threshold: DataCollector.add_failure()
4. When failures >= min_failure_cases: LoRATrainer.train()
5. Save checkpoint → ModelManager.load_lora_adapter(fuse=True)
6. Improved draft model → higher acceptance → faster inference
```

### Training Data Structure

Failures stored as JSONL in `data/failures/`:

```json
{
  "id": "unique_id",
  "prompt": "original prompt",
  "target_output": "what target model generated",
  "acceptance_rate": 0.35,
  "timestamp": "ISO-8601",
  "disagreements": [
    {
      "position": 12,
      "draft_token": 1234,
      "target_token": 5678,
      "draft_confidence": 0.45,
      "target_confidence": 0.78,
      "context_tokens": [...]
    }
  ]
}
```

## Testing Strategy

Tests validate critical components:

- **`test_lora_fusion.py`**: LoRA weight fusion correctness (regular and quantized layers)
- **`test_kv_cache.py`**: KV cache operations (creation, extension, rewinding)
- **`test_token_level_disagreement.py`**: Disagreement capture and serialization
- **`test_data_collector.py`**: Failure case collection and replay buffer
- **`test_eos_token_handling.py`**: End-of-sequence token handling

When modifying LoRA or cache logic, run corresponding tests immediately.

## Common Pitfalls

### LoRA and Quantization

- **Issue**: Trying to train on quantized layers without dequantizing first
  - **Solution**: `LoRALinear.__init__` handles this automatically via `mx.dequantize()`
- **Issue**: Loading LoRA adapter with wrong `fuse` flag
  - **Solution**: Use `fuse=True` for inference, `fuse=False` for continued training
- **Issue**: Forgetting to call `mx.eval()` after loading models
  - **Solution**: `ModelManager.load_models()` does this automatically

### KV Cache

- **Issue**: Not rewinding cache after rejecting draft tokens
  - **Solution**: Always call `rewind_cache()` before updating with accepted tokens
- **Issue**: Cache size mismatch errors
  - **Solution**: Check `get_cache_length()` before operations; respect `max_kv_cache_tokens`

### Memory Management

- **Issue**: Out of memory during training
  - **Solution**: Reduce `training.lora.rank`, enable `gradient_checkpointing`, lower batch size
- **Issue**: Slow model loading on first run
  - **Solution**: Expected behavior - models download to `~/.cache/huggingface/hub/`

### Data Collection

- **Issue**: Not collecting enough failure cases for effective training
  - **Solution**: Run with `--mode detailed` and lower `acceptance_threshold` temporarily
- **Issue**: Training on only failure cases causes catastrophic forgetting
  - **Solution**: System automatically mixes replay buffer (20% by default) - ensure it's enabled

## Platform Requirements

This project requires:
- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.10+**
- **16GB+ RAM** (32GB recommended for 7B target model)
- **MLX framework** (Apple's ML framework for unified memory architecture)

MLX-specific features used:
- Unified memory (CPU/GPU share same memory)
- 4-bit quantization for models
- Lazy tensor evaluation with `mx.eval()`
- Efficient KV cache operations
