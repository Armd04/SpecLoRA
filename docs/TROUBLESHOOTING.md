# Troubleshooting

Common issues and how to fix them.

## Installation Issues

### MLX Not Using GPU

**Symptom:**
```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# Output: Device(cpu, 0)
```

**Fix:**
You're not on Apple Silicon or MLX isn't installed correctly.

- Check you're on macOS with M1/M2/M3/M4: `uname -m` should say `arm64`
- Reinstall MLX: `pip install --upgrade mlx mlx-lm`
- Make sure you're in the venv: `which python` should point to `venv/bin/python`

### Missing Dependencies

**Symptom:**
```
ModuleNotFoundError: No module named 'mlx'
```

**Fix:**
Activate virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Version Conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Fix:**
Start fresh:
```bash
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Memory Issues

### Out of Memory During Training

**Symptom:**
```
RuntimeError: Failed to allocate memory
```

**Fix options (try in order):**

1. **Enable gradient checkpointing** (if not already):
   ```yaml
   # configs/config.yaml
   memory:
     gradient_checkpointing: true
   ```

2. **Reduce LoRA rank**:
   ```yaml
   training:
     lora:
       rank: 4  # Lower from 8
   ```

3. **Reduce sequence length**:
   ```yaml
   training:
     max_sequence_length: 1024  # Lower from 2048
   ```

4. **Use smaller target model**:
   ```yaml
   models:
     target:
       name: "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
   ```

5. **Reduce gradient accumulation**:
   ```yaml
   training:
     gradient_accumulation_steps: 2  # Lower from 4
   ```

### Out of Memory During Generation

**Symptom:**
```
RuntimeError: Failed to allocate memory for KV cache
```

**Fix:**
1. **Reduce max tokens**:
   ```yaml
   generation:
     max_tokens: 256  # Lower from 512
   ```

2. **Reduce KV cache size**:
   ```yaml
   models:
     target:
       max_kv_cache_tokens: 2048  # Lower from 4096
   ```

3. **Clear cache more often**:
   ```yaml
   memory:
     cache_clear_frequency: 5  # Lower from 10
   ```

### Memory Leak Over Time

**Symptom:**
Memory usage keeps growing over many generations.

**Fix:**
This is normal with MLX's lazy evaluation. The cache clearing should handle it, but if it doesn't:

```bash
# Restart the process periodically
# Or reduce cache_clear_frequency to 1
```

## Model Loading Issues

### Slow First Run

**Symptom:**
First run takes 10+ minutes, downloading models.

**Fix:**
This is normal. Models are ~15GB total and download to `~/.cache/huggingface/hub/`. Subsequent runs are fast.

Check download progress:
```bash
ls -lh ~/.cache/huggingface/hub/
```

### Model Download Fails

**Symptom:**
```
OSError: Can't load model from 'mlx-community/...'
```

**Fix:**
1. Check internet connection
2. Try downloading manually:
   ```bash
   huggingface-cli download mlx-community/Qwen2.5-3B-Instruct-4bit
   ```
3. If HuggingFace is down, wait and retry

### Tokenizer Mismatch

**Symptom:**
```
ValueError: Draft and target models must use the same tokenizer
```

**Fix:**
Use compatible model pairs. Qwen2.5 models share tokenizers:
- Target: Qwen2.5-3B or 7B
- Draft: Qwen2.5-0.5B

Don't mix model families (e.g., Qwen + Llama won't work).

## Performance Issues

### Low Acceptance Rate

**Symptom:**
Acceptance rates consistently below 40%.

**Possible causes:**

1. **Draft and target models too different**
   - Fix: Use models from same family (both Qwen2.5)

2. **Temperature too high**
   - Fix: Lower temperature:
     ```yaml
     speculative:
       temperature: 0.5  # Lower from 0.7
     ```

3. **Prompts outside training distribution**
   - Fix: Collect failures and train:
     ```bash
     python -m src.main train
     ```

4. **Draft model needs training**
   - Fix: Run with `--mode detailed` to collect more data

### Slower Than Expected

**Symptom:**
Not seeing 1.5-2x speedup.

**Check:**

1. **What's the acceptance rate?**
   ```bash
   python -m src.main generate "test" --mode fast
   # Look for "Acceptance Rate: X%"
   ```
   - Below 50%: Train the draft model
   - Above 60%: Should see good speedup

2. **Are you using detailed mode?**
   - Detailed mode is ~10-20% slower due to tracking overhead
   - Use `--mode fast` for production

3. **Is num_draft_tokens too high?**
   - If acceptance is low but K is high, you waste time on rejections
   - Lower K to 2-3 if acceptance < 50%

4. **Memory pressure?**
   - Check Activity Monitor for memory pressure
   - Close other apps

### Training Doesn't Improve Acceptance

**Symptom:**
Trained model, acceptance rate unchanged or worse.

**Debug steps:**

1. **Verify adapter loaded**:
   ```bash
   python -m src.main interactive
   # Type: /adapter-info
   ```
   Should show adapter is loaded.

2. **Check training loss**:
   ```bash
   cat data/checkpoints/best/trainer_state.json
   ```
   Loss should decrease. If it's NaN or increasing, training failed.

3. **Inspect failure cases**:
   ```bash
   head -5 data/failures/failures.jsonl
   ```
   Are they actually failures, or is data corrupted?

4. **Check replay buffer**:
   ```yaml
   training:
     replay_ratio: 0.2  # Make sure this isn't 0.0
   ```
   Without replay buffer, model may catastrophically forget.

5. **Train longer**:
   ```yaml
   training:
     num_epochs: 5  # Increase from 3
     min_failure_cases: 100  # Collect more data
   ```

## Data Collection Issues

### No Failure Cases Collected

**Symptom:**
```bash
ls data/failures/
# Empty or very few files
```

**Fix:**

1. **Lower acceptance threshold**:
   ```yaml
   speculative:
     acceptance_threshold: 0.7  # Raise from 0.5
   ```
   This marks more cases as failures.

2. **Use detailed mode**:
   ```bash
   python -m src.main generate "test" --mode detailed
   ```

3. **Run more generations**:
   ```bash
   python -m src.main collect-data --prompts-file prompts.txt
   ```

### Corrupted JSONL Files

**Symptom:**
```
json.decoder.JSONDecodeError: Expecting value
```

**Fix:**
Check file integrity:
```bash
cat data/failures/failures.jsonl | jq .
```

If corrupted, delete and re-collect:
```bash
rm data/failures/failures.jsonl
python -m src.main collect-data --prompts-file prompts.txt
```

## Training Issues

### NaN Loss During Training

**Symptom:**
```
WARNING: NaN or Inf detected in loss
```

**Fix:**

1. **Lower learning rate**:
   ```yaml
   training:
     learning_rate: 5.0e-5  # Lower from 1e-4
   ```

2. **Check for invalid tokens**:
   Training data may contain tokens outside vocab. The system should filter these, but verify:
   ```bash
   python -c "
   import json
   with open('data/failures/failures.jsonl') as f:
       for line in f:
           ex = json.loads(line)
           print(max(ex['prompt_tokens']), max(ex['target_output']['tokens']))
   "
   ```
   All values should be < vocab_size (~151936 for Qwen).

3. **Reduce batch size** (via gradient accumulation):
   ```yaml
   training:
     gradient_accumulation_steps: 2  # Lower from 4
   ```

### Training Crashes

**Symptom:**
Process killed during training.

**Fix:**
Out of memory. See "Out of Memory During Training" section above.

### Can't Load Checkpoint

**Symptom:**
```
FileNotFoundError: data/checkpoints/best/adapters.safetensors
```

**Fix:**
Training never completed or checkpoint is corrupt.

1. **Check if training finished**:
   ```bash
   cat data/checkpoints/best/trainer_state.json
   ```
   Look for `global_step > 0`.

2. **If corrupt, delete and retrain**:
   ```bash
   rm -rf data/checkpoints/best/
   python -m src.main train
   ```

## Runtime Issues

### EOS Token Errors

**Symptom:**
```
ValueError: EOS token not found in tokenizer
```

**Fix:**
This shouldn't happen with Qwen models. If it does:

1. **Check tokenizer**:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
   print(tok.eos_token_id)  # Should be 151645
   ```

2. **Update transformers**:
   ```bash
   pip install --upgrade transformers
   ```

### KV Cache Corruption

**Symptom:**
Outputs become gibberish after several generations.

**Fix:**
Cache wasn't rewound properly after rejection. This is a bug.

Workaround:
```bash
# Restart the process
# Or reduce cache_clear_frequency to 1
```

Report this as an issue with:
- The prompt that triggered it
- Acceptance rate at the time
- Full error output

## Testing Issues

### Tests Fail

**Symptom:**
```bash
python -m pytest tests/
# FAILED tests/...
```

**Fix:**

1. **Check you're in venv**:
   ```bash
   which python  # Should be in venv/bin/
   ```

2. **Install dev dependencies**:
   ```bash
   pip install pytest
   ```

3. **Run specific failing test**:
   ```bash
   python -m pytest tests/test_lora_fusion.py::test_fusion_correctness -v
   ```

4. **If test is actually broken**, report it with:
   - Test name
   - Full error output
   - Your environment (`python --version`, `uname -m`)

## Still Stuck?

1. **Check logs**: Look in terminal output for detailed error messages
2. **Enable debug logging**:
   ```yaml
   # configs/config.yaml
   logging:
     level: "DEBUG"
   ```
3. **Verify environment**:
   ```bash
   python --version  # Should be 3.10+
   uname -m          # Should be arm64
   python -c "import mlx.core as mx; print(mx.default_device())"  # Should be gpu
   ```
4. **Open an issue** on GitHub with:
   - What you tried to do
   - Full error message
   - Config file
   - Environment details

## Performance Metrics Explained

### Acceptance Rate

**What it means:** Percentage of draft tokens accepted by target model

**Good values:**
- 60-80%: Excellent, 1.5-2x speedup
- 40-60%: OK, some speedup
- <40%: Poor, train draft model

**How to improve:**
- Train on failure cases
- Lower temperature
- Reduce num_draft_tokens

### Tokens/Second

**What it means:** Generation speed

**Typical values:**
- Target only: ~30-50 tokens/sec (baseline)
- With speculative (60% acceptance): ~50-80 tokens/sec
- With speculative (80% acceptance): ~70-100 tokens/sec

**How to improve:**
- Improve acceptance rate (see above)
- Use greedy decoding (temperature=0)
- Close other apps to free memory

### Training Loss

**What it means:** How well draft model predicts target outputs

**Good values:**
- Decreasing over time
- Final value < 1.0 usually good
- Not NaN or Inf

**Red flags:**
- Increasing loss: Overfitting or bad data
- NaN/Inf: Numerical instability
- No change: Learning rate too low or data issue
