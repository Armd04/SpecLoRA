# Benchmark Suite Documentation

## Overview

The SpecLoRA benchmark suite provides comprehensive performance comparison across three decoding modes:

1. **Target Only** - Standard autoregressive decoding with the target model (baseline)
2. **Spec Dec (Base)** - MLX-LM speculative decoding with unmodified draft model
3. **Spec Dec (LoRA)** - MLX-LM speculative decoding with LoRA-adapted draft model

## Usage

### Basic Commands

```bash
# Run with config evaluation prompts (no LoRA)
python -m src.main benchmark-suite --skip-lora

# Use prompts from file
python -m src.main benchmark-suite -f prompts.txt --limit 15

# Run multiple iterations for averaging
python -m src.main benchmark-suite -f prompts.txt -n 3

# Export results to JSON
python -m src.main benchmark-suite -f prompts.txt -o results/benchmark.json

# Use specific LoRA adapter
python -m src.main benchmark-suite -a data/checkpoints/final
```

### Command Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--prompts-file` | `-f` | config prompts | File with prompts (one per line) |
| `--limit` | `-l` | None | Limit number of prompts (randomly samples) |
| `--max-tokens` | `-m` | 256 | Maximum tokens per generation |
| `--iterations` | `-n` | 1 | Iterations per prompt for averaging |
| `--lora-adapter` | `-a` | best | Adapter path or alias (best/final/latest) |
| `--skip-lora` | - | False | Skip LoRA benchmarks |
| `--output` | `-o` | None | JSON output file path |

## Metrics Collected

### Per-Prompt Metrics

- **Tokens/Second** - Generation speed (higher is better)
- **Time to Last Token (TTLT)** - Total generation time in seconds
- **Acceptance Rate** - Draft token acceptance percentage (spec dec only)
- **Total Tokens** - Number of tokens generated

### Aggregate Statistics

- **Average Tokens/Second** - Mean across all prompts
- **Average TTLT** - Mean time to last token
- **Average Acceptance Rate** - Mean acceptance rate (spec dec modes)
- **Speedup vs Target** - Relative speedup compared to baseline
- **LoRA Improvement** - Performance gain from LoRA adapter

## Output Format

### Console Output

The benchmark displays results in rich console tables:

```
╭──────────────────────────────────────────────────────────────────╮
│                  SpecLoRA Benchmark Summary                       │
├──────────────────┬──────────────┬──────────┬────────────────────┤
│ Mode             │ Avg Tok/s    │ Avg TTLT │ Avg Acceptance     │
├──────────────────┼──────────────┼──────────┼────────────────────┤
│ Target Only      │ 45.2         │ 3.21s    │ -                  │
│ Spec Dec (Base)  │ 62.3         │ 2.35s    │ 58.2%              │
│ Spec Dec (LoRA)  │ 71.8         │ 2.01s    │ 72.1%              │
├──────────────────┴──────────────┴──────────┴────────────────────┤
│ Speedup vs Target: Base=1.38x, LoRA=1.59x                        │
│ LoRA improvement over Base: +15.2% tok/s, +13.9% acceptance      │
╰──────────────────────────────────────────────────────────────────╯
```

### JSON Export

When using `--output`, results are exported in JSON format:

```json
{
  "metadata": {
    "timestamp": "2024-12-24T10:30:00",
    "num_prompts": 15,
    "max_tokens": 256,
    "iterations": 3,
    "adapter_path": "data/checkpoints/best"
  },
  "summary": {
    "target_only": {
      "mode": "target_only",
      "avg_tokens_per_second": 45.2,
      "avg_time_to_last_token": 3.21,
      "avg_acceptance_rate": null,
      "total_tokens": 3840,
      "num_prompts": 15,
      "min_tokens_per_second": 38.5,
      "max_tokens_per_second": 52.1
    },
    "spec_base": {
      "mode": "spec_base",
      "avg_tokens_per_second": 62.3,
      "avg_time_to_last_token": 2.35,
      "avg_acceptance_rate": 0.582,
      "total_tokens": 3840,
      "num_prompts": 15,
      "min_tokens_per_second": 55.2,
      "max_tokens_per_second": 68.9
    },
    "spec_lora": {
      "mode": "spec_lora",
      "avg_tokens_per_second": 71.8,
      "avg_time_to_last_token": 2.01,
      "avg_acceptance_rate": 0.721,
      "total_tokens": 3840,
      "num_prompts": 15,
      "min_tokens_per_second": 64.3,
      "max_tokens_per_second": 79.2
    }
  },
  "per_prompt": [
    {
      "prompt": "What is Python?",
      "full_prompt": "What is Python?",
      "target_only": {
        "tokens_per_second": 44.5,
        "time_to_last_token": 3.42,
        "acceptance_rate": null,
        "total_tokens": 152
      },
      "spec_base": {
        "tokens_per_second": 61.2,
        "time_to_last_token": 2.48,
        "acceptance_rate": 0.56,
        "total_tokens": 152
      },
      "spec_lora": {
        "tokens_per_second": 70.8,
        "time_to_last_token": 2.15,
        "acceptance_rate": 0.71,
        "total_tokens": 152
      }
    },
    ...
  ]
}
```

## Example Workflows

### Quick Test (No LoRA)

Test the benchmark suite without needing a trained adapter:

```bash
python -m src.main benchmark-suite --skip-lora --max-tokens 128
```

### Full Benchmark with Training

Complete workflow including training and benchmarking:

```bash
# 1. Collect training data
python -m src.main collect-data -f prompts.txt --limit 50

# 2. Train the draft model
python -m src.main train

# 3. Run benchmark suite
python -m src.main benchmark-suite -f prompts.txt --limit 20 -o results.json

# 4. Analyze results
cat results.json | python -m json.tool
```

### Comparing Multiple Adapters

To compare different training checkpoints:

```bash
# Benchmark with 'best' checkpoint
python -m src.main benchmark-suite -f test_prompts.txt -a best -o best_results.json

# Benchmark with 'final' checkpoint
python -m src.main benchmark-suite -f test_prompts.txt -a final -o final_results.json

# Compare results
diff best_results.json final_results.json
```

### Averaging Multiple Runs

For more stable results, run multiple iterations:

```bash
python -m src.main benchmark-suite -f prompts.txt --limit 10 -n 5 -o averaged_results.json
```

This runs each prompt 5 times and averages the metrics.

## Tips for Effective Benchmarking

### Prompt Selection

- **Diversity**: Use prompts covering different tasks (reasoning, coding, factual)
- **Length**: Mix short and long prompts to test different scenarios
- **Difficulty**: Include both easy and hard prompts for the draft model

### Number of Prompts

- **Quick Test**: 5-10 prompts (2-3 minutes)
- **Standard Benchmark**: 15-20 prompts (5-10 minutes)
- **Comprehensive**: 50+ prompts (20-30 minutes)

### Iterations

- **Single Run**: Fast but variable results
- **3 Iterations**: Good balance of speed and stability
- **5+ Iterations**: Most stable, use for final evaluations

### When to Use --skip-lora

Use `--skip-lora` when:
- No adapter has been trained yet
- Only interested in base speculative decoding performance
- Quickly testing the benchmark suite setup

## Interpreting Results

### Expected Speedups

Good benchmarks typically show:
- **Spec Dec (Base) vs Target**: 1.2-1.5x speedup
- **Spec Dec (LoRA) vs Target**: 1.5-2.0x speedup
- **LoRA vs Base**: 10-30% improvement

### Acceptance Rates

- **<50%**: Draft model struggles, may need more training
- **50-70%**: Reasonable performance, room for improvement
- **70-85%**: Good performance, effective LoRA adaptation
- **>85%**: Excellent performance, diminishing returns

### When LoRA Helps Most

LoRA adapters provide the biggest gains when:
- Base acceptance rate is between 40-70%
- Training data covers similar tasks to benchmark prompts
- Draft model has been trained on enough failure cases (30+)

## Troubleshooting

### "No adapter found" Error

```bash
# Check available adapters
ls -la data/checkpoints/

# Verify adapter structure
python -c "from src.main import SpeculativeDecodingSystem; import yaml; config = yaml.safe_load(open('configs/config.yaml')); s = SpeculativeDecodingSystem(config); s.initialize(); print(s.resolve_adapter_path('best'))"
```

### Slow Benchmarks

If benchmarks are running slowly:
- Reduce `--max-tokens` (default 256)
- Reduce `--limit` to test fewer prompts
- Use `--skip-lora` to skip the third phase

### Out of Memory

For 16GB MacBooks:
- Use shorter prompts
- Reduce `--max-tokens` to 128 or less
- Ensure `cache_clear_frequency` is set in config.yaml

## Files and Directories

### Created Files

- `src/benchmark.py` - Benchmark module
- `examples/benchmark_test_prompts.txt` - Test prompts

### Modified Files

- `src/main.py` - Added `benchmark-suite` CLI command
- `configs/config.yaml` - Added benchmark configuration section

### Output Locations

- JSON results: Specified by `--output` or default to stdout
- Logs: `data/training.log` (if logging enabled)

## API Reference

See `src/benchmark.py` for detailed API documentation of:
- `BenchmarkResult` - Single benchmark result dataclass
- `ModeSummary` - Aggregate statistics per mode
- `BenchmarkSummary` - Complete benchmark summary
- `BenchmarkRunner` - Core benchmark execution class
- `run_benchmark_suite()` - Main entry point function

