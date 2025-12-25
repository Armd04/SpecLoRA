# Architecture

This document explains how SpecLoRA works internally.

## Core Components

### Model Manager (`src/models.py`)

Handles loading and managing models:

- Loads target and draft models from HuggingFace with MLX
- Supports both regular and 4-bit quantized models
- Manages LoRA adapter loading with two modes:
  - `fuse=True`: Merge LoRA weights into base model (inference)
  - `fuse=False`: Keep LoRA separate as trainable layers (training)
- Provides KV cache utilities for efficient attention

**Key functions:**
- `load_models()`: Loads target and draft models
- `load_lora_adapter()`: Loads LoRA weights and fuses or wraps layers
- `create_kv_cache()`, `get_logits_with_cache()`, `rewind_cache()`: KV cache operations

### Two-Mode Speculative Decoding

The system operates in two distinct modes:

#### Fast Mode (`src/speculative.py`)

Production mode optimized for speed:

- Wraps MLX-LM's built-in speculative decoding
- Minimal overhead
- Tracks overall acceptance rates only
- Returns `SpeculativeResult` with text, tokens, and metrics

Use this for actual inference when you care about speed.

#### Detailed Mode (`src/speculative_manual.py`)

Research/training mode with full tracking:

- Custom implementation that captures token-level disagreements
- Records every draft/target mismatch with confidence scores
- Comparable speed to fast mode via aggressive KV caching
- Returns `ManualSpeculativeResult` with detailed disagreement data
- Essential for collecting training data

Use this when running `--mode detailed` to gather failure cases.

### Data Collector (`src/data_collector.py`)

Manages failure case collection and replay buffer:

**Data structures:**
- `TrainingExample`: Stores prompt, target output, acceptance rate, disagreements
- `TokenLevelDisagreement`: Position, draft/target tokens, confidence scores
- `AcceptanceRateTracker`: Rolling statistics on acceptance rates

**Components:**
- `DataCollector`: Saves failures to JSONL in `data/failures/`
- Replay buffer: Maintains successful cases to prevent catastrophic forgetting
- Stores prompt tokens to avoid tokenizer drift during training

### LoRA Trainer (`src/training.py`)

Handles fine-tuning the draft model:

**LoRALinear Layer:**
- Wraps base Linear or QuantizedLinear layers
- Adds low-rank matrices A and B
- Supports dequantization for quantized models during training
- Forward pass: `output = base_layer(x) + (x @ A) @ B`

**LoRATrainer:**
- Loads failure cases and mixes with replay buffer (80/20 split by default)
- Trains only LoRA parameters (~0.5% of model size)
- Gradient accumulation for memory efficiency
- Learning rate warmup and cosine decay
- Saves MLX-LM compatible checkpoints to `data/checkpoints/`

**Training flow:**
1. Load draft model (dequantize if quantized)
2. Wrap target layers with LoRALinear
3. Train on mixed batch (failures + replay)
4. Save LoRA weights
5. For inference: load with `fuse=True` to merge weights

### CLI Orchestration (`src/main.py`)

The `SpeculativeDecodingSystem` class coordinates everything:

**Commands:**
- `generate`: Generate text with speculative decoding
- `train`: Train draft model on collected failures
- `interactive`: Interactive mode with `/stats`, `/train`, `/eval` commands
- `demo`: Run demo to showcase the system
- `collect-data`: Batch data collection from prompts file
- `benchmark`: Performance benchmarking

**Responsibilities:**
- Loads config from `configs/config.yaml`
- Initializes models and decoders
- Routes commands to appropriate components
- Manages cache clearing every N generations
- Tracks generation count for triggering training

## Data Flow

### Inference → Collection → Training Loop

```
1. User prompt
   ↓
2. Format with tokenizer.apply_chat_template()
   ↓
3. Generate with SpeculativeDecoder (fast) or ManualSpeculativeDecoder (detailed)
   ↓
4. If acceptance_rate < threshold:
   → DataCollector.add_failure()
   ↓
5. When failures >= min_failure_cases:
   → LoRATrainer.train()
   ↓
6. Save checkpoint → ModelManager.load_lora_adapter(fuse=True)
   ↓
7. Improved draft model → higher acceptance → faster inference
```

### Speculative Decoding Algorithm

Each generation step:

```
1. Draft model generates K tokens (e.g., K=4)
   - Uses KV cache from previous accepted tokens
   - Returns: [token1, token2, token3, token4]

2. Target model verifies all K tokens in parallel
   - Single forward pass with all K tokens
   - Computes probability for each position
   - Returns: acceptance decisions + probabilities

3. Process results
   - Accept tokens while draft matches target
   - On first rejection:
     * Resample that position from target distribution
     * Discard remaining draft tokens

4. Update KV caches
   - Rewind both caches to before draft tokens
   - Extend caches with only accepted tokens
   - Ready for next iteration

5. Return when:
   - Generated max_tokens
   - EOS token sampled
   - Max KV cache size reached
```

**Why it's fast:** When draft model is accurate, you get K tokens for ~1.5x the cost of generating 1 token normally (draft pass + verification pass).

## LoRA Fusion and Quantization

Understanding the full lifecycle:

### Training Phase

1. Load draft model (may be 4-bit quantized)
2. If quantized, dequantize weights: `W_fp16 = mx.dequantize(W_4bit)`
3. Wrap with LoRALinear: keeps base frozen, only A and B are trainable
4. Train on mixed batch (failures + replay buffer)
5. Save LoRA matrices A and B to checkpoint

### Inference Phase (Fusion)

1. Load draft model
2. Load LoRA adapter with `fuse=True`
3. For each wrapped layer: `W_new = W_base + (A @ B)`
4. Replace layer with single Linear layer containing fused weights
5. Optionally re-quantize fused weights for memory savings

**Result:** Zero runtime overhead from LoRA during inference.

### Continued Training

1. Load existing checkpoint with `fuse=False`
2. Layers remain wrapped as LoRALinear
3. A and B stay separate and trainable
4. Continue training from previous state

## KV Cache Management

Critical for performance in speculative decoding:

**What is KV cache?**
- Stores Key and Value tensors from attention layers
- Avoids recomputing attention for previous tokens
- Format: list of (K, V) tuples, one per layer

**Operations:**

```python
# Create empty cache
cache = create_kv_cache(model, max_tokens)

# Extend cache during generation
logits = get_logits_with_cache(model, tokens, cache)

# Rewind cache after rejecting draft tokens
rewind_cache(cache, new_length)

# Check cache size
current_size = get_cache_length(cache)
```

**Why rewinding matters:**

During speculative decoding, we speculatively extend the cache with draft tokens. If those tokens get rejected, we must rewind the cache to maintain correctness.

Example:
```
1. Cache contains tokens [1, 2, 3]  (length = 3)
2. Draft generates [4, 5, 6, 7]
3. Speculatively extend cache to [1, 2, 3, 4, 5, 6, 7]  (length = 7)
4. Target verifies: accepts [4, 5], rejects [6, 7]
5. Rewind cache to length 5: [1, 2, 3, 4, 5]
6. Continue from token 5
```

Without rewinding, the cache would contain invalid entries and generate garbage.

## Memory Management

SpecLoRA is designed to run on 16GB MacBooks:

**Strategies:**
1. **4-bit quantization**: Models use ~25% of FP16 memory
2. **LoRA training**: Only 0.5% of parameters trainable
3. **Gradient checkpointing**: Recompute activations during backward pass
4. **Cache clearing**: MLX cache cleared every N generations
5. **Batch size = 1**: No batching during training/inference

**Memory breakdown (16GB machine):**
- Target model (3B-4bit): ~2GB
- Draft model (0.5B-4bit): ~300MB
- KV caches: ~1-2GB (depends on sequence length)
- Activations during training: ~2-4GB
- OS and other processes: ~8GB
- Remaining: ~2GB buffer

For 32GB+ machines, you can use 7B target models and higher LoRA ranks.

## Testing Strategy

Tests validate critical invariants:

**LoRA fusion (`test_lora_fusion.py`):**
- LoRALinear output matches fused Linear output
- Identity initialization (A=0 or B=0 → no change)
- MLX-LM checkpoint format compatibility
- Both regular and quantized layer support

**KV cache (`test_kv_cache.py`):**
- Cache creation and extension
- Rewinding maintains correctness
- Length tracking accurate

**Data collection (`test_data_collector.py`):**
- Failure case serialization/deserialization
- Replay buffer management
- Token-level disagreement capture

**Adapter loading (`test_adapter_loading.py`):**
- Path resolution (aliases, relative, absolute)
- Validation and error handling
- Rollback on failures
- Interactive commands

Run tests before modifying core components:
```bash
python -m pytest tests/test_lora_fusion.py  # Before changing LoRA code
python -m pytest tests/test_kv_cache.py     # Before changing cache logic
```
