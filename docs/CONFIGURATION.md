# Configuration Guide

All settings live in `configs/config.yaml`. This guide explains what each option does and how to tune them.

## Model Configuration

```yaml
models:
  target:
    name: "mlx-community/Qwen2.5-3B-Instruct-4bit"
    max_kv_cache_tokens: 4096
  draft:
    name: "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    max_kv_cache_tokens: 4096
```

**target.name**: The large, accurate model used for verification
- **Qwen2.5-3B-Instruct-4bit**: Good for 16GB RAM (default)
- **Qwen2.5-7B-Instruct-4bit**: Needs 32GB+ RAM, better quality
- Must be quantized (4-bit) to fit in memory

**draft.name**: The small, fast model used for drafting
- **Qwen2.5-0.5B-Instruct-4bit**: Only option that's fast enough (default)
- Must match target model's tokenizer

**max_kv_cache_tokens**: Maximum sequence length for KV cache
- Default 4096 is fine for most use cases
- Increase if you need longer contexts (uses more memory)
- Decrease if running out of memory

## Speculative Decoding

```yaml
speculative:
  num_draft_tokens: 4
  temperature: 0.7
  top_p: 0.9
  acceptance_threshold: 0.5
```

**num_draft_tokens** (K): How many tokens draft model generates per iteration
- **Default: 4** - good balance of speed and acceptance
- Lower (2-3): Higher acceptance rate, less speedup
- Higher (6-8): More speedup if acceptance stays high, but risky
- Sweet spot is usually 4-5

**temperature**: Sampling randomness
- **0.0**: Greedy (deterministic), highest acceptance rates
- **0.7**: Balanced (default)
- **1.0+**: Creative but lower acceptance rates

**top_p**: Nucleus sampling cutoff
- **0.9**: Default, good balance
- Lower: More deterministic
- Higher: More diverse

**acceptance_threshold**: Below this rate, case is marked as failure
- **0.5** (50%): Default - collect cases where draft gets <50% right
- Lower (0.3): Only collect really bad cases
- Higher (0.7): More aggressive training, more data

## Chat Formatting

```yaml
chat:
  system_message: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
```

**system_message**: The system prompt prepended to all user prompts
- Uses tokenizer's native `apply_chat_template()` method
- Only the system message is configurable
- Different models expect different formats (handled automatically)

## LoRA Training

```yaml
training:
  lora:
    rank: 8
    alpha: 16
    target_modules: ["q_proj", "v_proj"]
  learning_rate: 1.0e-4
  batch_size: 1
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_steps: 10
  min_failure_cases: 50
  replay_ratio: 0.2
  max_sequence_length: 2048
```

### LoRA Parameters

**rank**: Size of low-rank matrices (r in the paper)
- **Default: 8** - works on 16GB RAM
- Lower (4): Less memory, less capacity
- Higher (16, 32): More capacity, needs more RAM
- Typical range: 4-64

**alpha**: Scaling factor for LoRA updates
- **Default: 16** (usually 2x rank)
- Higher: Stronger updates from LoRA
- Lower: Weaker updates
- Formula: `output = base(x) + (alpha/rank) * lora(x)`

**target_modules**: Which layers to wrap with LoRA
- **Default: ["q_proj", "v_proj"]** - query and value projections in attention
- More modules: Better but slower training
- Options: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Start with q_proj and v_proj, add more if needed

### Training Hyperparameters

**learning_rate**: How fast to update weights
- **Default: 1e-4** - conservative, stable
- Higher (5e-4): Faster learning, may be unstable
- Lower (5e-5): Slower, more stable
- Use warmup to ramp up gradually

**batch_size**: Examples per batch
- **Default: 1** - only option for 16GB RAM
- Higher: Better gradients, needs more memory

**gradient_accumulation_steps**: Simulate larger batches
- **Default: 4** - effective batch size = 4
- Accumulates gradients over N steps before updating
- Memory-efficient way to get larger batch effects
- Increase to 8-16 for more stable training

**num_epochs**: How many times to loop through training data
- **Default: 3** - prevents overfitting
- More: Risk overfitting on small datasets
- Less: May underfit

**warmup_steps**: Steps to ramp up learning rate from 0 to max
- **Default: 10** - gentle warmup
- Prevents instability at training start
- Usually 5-10% of total steps

### Data Collection

**min_failure_cases**: Minimum failures before training triggers
- **Default: 50** - enough for meaningful training
- Lower (20): Train sooner, but noisier
- Higher (100+): More data, slower feedback loop

**replay_ratio**: Fraction of each batch from replay buffer
- **Default: 0.2** (20%) - prevents catastrophic forgetting
- 0.0: Only train on failures (risky, may forget good behavior)
- 0.5: Equal mix of failures and successes
- Higher: More conservative, slower improvement

**max_sequence_length**: Truncate sequences longer than this
- **Default: 2048** - handles most prompts
- Longer: More memory usage
- Shorter: Saves memory, may cut off long examples

## Memory Configuration

```yaml
memory:
  gradient_checkpointing: true
  cache_clear_frequency: 10
```

**gradient_checkpointing**: Trade computation for memory
- **Default: true** - essential for 16GB RAM
- Recomputes activations during backward pass
- ~30% slower training, ~50% less memory
- Disable only if you have 32GB+ RAM

**cache_clear_frequency**: Clear MLX cache every N generations
- **Default: 10** - prevents cache buildup
- MLX caches computation graphs
- Higher: Better performance, may leak memory
- Lower: More conservative, slight overhead

## Generation Settings

```yaml
generation:
  max_tokens: 512
  min_tokens: 1
```

**max_tokens**: Maximum tokens to generate
- **Default: 512** - reasonable for most responses
- Increase for longer outputs
- Each token uses more KV cache memory

**min_tokens**: Minimum tokens before allowing EOS
- **Default: 1** - no minimum
- Prevents very short responses

## Logging

```yaml
logging:
  level: "INFO"
  tensorboard: false
```

**level**: How verbose the logs are
- **INFO**: Default, balanced
- **DEBUG**: Very verbose, useful for troubleshooting
- **WARNING**: Only warnings and errors

**tensorboard**: Enable TensorBoard logging
- **Default: false** - not implemented yet
- Future: Training curves, metrics visualization

## Tuning for Your Hardware

### 16GB RAM (M1/M2 MacBook)

```yaml
models:
  target:
    name: "mlx-community/Qwen2.5-3B-Instruct-4bit"
training:
  lora:
    rank: 8
  batch_size: 1
  gradient_accumulation_steps: 4
memory:
  gradient_checkpointing: true
```

This is the default config. Stick with it.

### 32GB+ RAM

```yaml
models:
  target:
    name: "mlx-community/Qwen2.5-7B-Instruct-4bit"  # Larger model
training:
  lora:
    rank: 16  # More capacity
  batch_size: 2  # Can fit 2 examples
  gradient_accumulation_steps: 2
memory:
  gradient_checkpointing: false  # Can disable for speed
```

### Prioritizing Speed

```yaml
speculative:
  num_draft_tokens: 6  # Draft more aggressively
  temperature: 0.0  # Greedy = higher acceptance
  acceptance_threshold: 0.3  # Only collect terrible cases
training:
  min_failure_cases: 100  # Train less frequently
```

### Prioritizing Quality

```yaml
speculative:
  num_draft_tokens: 3  # Conservative drafting
  temperature: 0.8  # More creative
  acceptance_threshold: 0.6  # Collect more cases
training:
  min_failure_cases: 30  # Train more frequently
  lora:
    rank: 16  # More capacity
```

## Performance Impact

| Parameter | Higher Value | Lower Value |
|-----------|--------------|-------------|
| num_draft_tokens | More speedup (if accepted) | Higher acceptance rate |
| temperature | More diverse, slower | Faster, more deterministic |
| lora.rank | Better quality, more memory | Less memory, faster training |
| learning_rate | Faster learning, unstable | Slower, more stable |
| min_failure_cases | Less frequent training | More frequent training |
| replay_ratio | Safer, slower improvement | Faster improvement, risky |

## Validation

The config is validated on load. Common errors:

**Invalid model name**: Model must exist on HuggingFace
**Mismatched tokenizers**: Draft and target must use same tokenizer
**Rank too high**: Out of memory during training
**Negative values**: Many parameters must be positive

Check logs for validation errors at startup.
