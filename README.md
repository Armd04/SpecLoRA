# SpecLoRA

**Make your language models faster by teaching them from their mistakes.**

SpecLoRA combines speculative decoding with adaptive LoRA training to speed up inference on Apple Silicon. Instead of just using a small draft model to guess what a larger model would say, the system learns from disagreements and improves over time.

## The Idea

Large language models are slow because they generate one token at a time. Speculative decoding speeds this up by having a small "draft" model quickly guess the next few tokens, then checking them all at once with the large "target" model. When the draft guesses correctly, you get multiple tokens for the price of one verification.

The problem? Draft models often disagree with target models, killing the speedup.

The solution: **Learn from disagreements.** When the draft model gets it wrong, we collect those cases and fine-tune it using LoRA (Low-Rank Adaptation). Over time, the draft model gets better at predicting what the target would say, acceptance rates go up, and everything gets faster.

## Quick Start

```bash
# Setup
git clone <repository-url>
cd SpecLoRA
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run demo
python -m src.main demo
```

The first run downloads models (~15GB). After that, you can:

```bash
# Generate text
python -m src.main generate "Explain quantum computing simply."

# Interactive mode
python -m src.main interactive

# Train on collected failures
python -m src.main train
```

## Requirements

- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.10+**
- **16GB+ RAM** (32GB better for larger models)

This uses [MLX](https://github.com/ml-explore/mlx), Apple's ML framework optimized for unified memory.

## How It Works

1. **Draft model** (Qwen2.5-0.5B) quickly generates 4 candidate tokens
2. **Target model** (Qwen2.5-3B) verifies all 4 in parallel
3. Accepted tokens are kept, rejected ones are resampled
4. When acceptance rate is low, the case gets saved
5. After collecting enough failures, train the draft model with LoRA
6. Improved draft → higher acceptance → faster inference

The feedback loop means the system gets faster the more you use it.

## Key Features

**Speculative Decoding**: 1.5-2x speedup out of the box
**Adaptive Training**: Draft model improves from failures
**Memory Efficient**: Runs on 16GB MacBooks with 4-bit quantization
**Two Modes**: Fast mode for production, detailed mode for data collection

## Configuration

Edit `configs/config.yaml` to change models, adjust how many tokens to draft (K), set the acceptance threshold for failures, or tune LoRA training parameters.

The defaults work well for 16GB machines. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for details.

## Project Structure

```
src/
  models.py            # Model loading, LoRA fusion, KV cache
  speculative.py       # Fast mode (wraps MLX-LM)
  speculative_manual.py  # Detailed mode (token-level tracking)
  data_collector.py    # Failure case collection
  training.py          # LoRA training pipeline
  main.py             # CLI commands

data/
  failures/           # Collected failure cases (JSONL)
  checkpoints/        # Trained LoRA adapters

configs/
  config.yaml         # All settings
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - How the system works internally
- **[Configuration](docs/CONFIGURATION.md)** - Settings and tuning guide
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Reference](docs/API.md)** - Using SpecLoRA programmatically

## Development

```bash
# Run tests
python -m pytest tests/

# Linting and formatting
ruff check . --fix
ruff format .

# Install pre-commit hooks
pre-commit install
```

## References

- [Speculative Decoding](https://arxiv.org/abs/2211.17192) - Original paper
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [MLX](https://github.com/ml-explore/mlx) - Apple's ML framework
- [Qwen2.5](https://huggingface.co/Qwen) - Models used in this project

## License

MIT
