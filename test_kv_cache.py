#!/usr/bin/env python3
"""
Simple test script to verify KV caching in manual speculative decoding.
"""

import logging
from src.models import ModelManager
from src.speculative_manual import ManualSpeculativeDecoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_manual_specdec_with_cache():
    """Test that manual speculative decoding works with KV caching."""
    logger.info("=" * 60)
    logger.info("Testing Manual Speculative Decoding with KV Cache")
    logger.info("=" * 60)

    # Use smaller models for faster testing
    target_model_name = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    draft_model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    logger.info(f"Target model: {target_model_name}")
    logger.info(f"Draft model: {draft_model_name}")

    # Load models
    logger.info("Loading models...")
    model_manager = ModelManager(
        target_model_name=target_model_name,
        draft_model_name=draft_model_name,
    )
    model_manager.load_models()

    target_model, tokenizer = model_manager.get_target_model()
    draft_model, _ = model_manager.get_draft_model()

    # Create decoder
    decoder = ManualSpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        num_draft_tokens=4,
        temperature=0.0,  # Greedy for deterministic results
    )

    # Test prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]

    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n--- Test {i+1}/{len(test_prompts)} ---")
        logger.info(f"Prompt: {prompt}")

        result = decoder.generate(prompt, max_tokens=50)

        logger.info(f"Generated: {result.text[:100]}...")
        logger.info(f"Tokens generated: {result.metrics.total_tokens_generated}")
        logger.info(f"Draft tokens proposed: {result.metrics.draft_tokens_proposed}")
        logger.info(f"Draft tokens accepted: {result.metrics.draft_tokens_accepted}")
        logger.info(f"Acceptance rate: {result.metrics.acceptance_rate:.1%}")
        logger.info(f"Speed: {result.metrics.tokens_per_second:.1f} tokens/s")
        logger.info(f"Disagreements: {len(result.disagreements)}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ Test completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    test_manual_specdec_with_cache()
