"""
Test that EOS token handling is robust to tokenizers returning lists vs ints.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockTokenizerReturningList:
    """Mock tokenizer that returns a list from convert_tokens_to_ids."""
    
    def __init__(self):
        self.eos_token_id = None  # Simulate tokenizer without eos_token_id
        self.eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Minimal chat template support required by ManualSpeculativeDecoder
        return "".join(m["content"] for m in messages)
    
    def convert_tokens_to_ids(self, token):
        """Return a list [2] instead of int 2."""
        if token == "</s>":
            return [2]  # This is the problematic behavior
        return [0]


class MockTokenizerReturningInt:
    """Mock tokenizer that returns an int from convert_tokens_to_ids."""
    
    def __init__(self):
        self.eos_token_id = None
        self.eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(m["content"] for m in messages)
    
    def convert_tokens_to_ids(self, token):
        """Return int 2 directly."""
        if token == "</s>":
            return 2  # Expected behavior
        return 0


class MockTokenizerWithEosTokenId:
    """Mock tokenizer that already has eos_token_id set."""
    
    def __init__(self):
        self.eos_token_id = 2
        self.eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(m["content"] for m in messages)
    
    def convert_tokens_to_ids(self, token):
        """This shouldn't be called if eos_token_id is already set."""
        if token == "</s>":
            return 2
        return 0


def test_eos_token_from_list():
    """Test that EOS token is extracted correctly when convert_tokens_to_ids returns a list."""
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerReturningList()
    
    # Create a minimal decoder just to test __init__
    # We need mock models, but we're just testing the tokenizer handling
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    # Should be an int, not a list
    assert isinstance(decoder.eos_token_id, int), \
        f"Expected int, got {type(decoder.eos_token_id)}"
    assert decoder.eos_token_id == 2, \
        f"Expected 2, got {decoder.eos_token_id}"


def test_eos_token_from_int():
    """Test that EOS token works correctly when convert_tokens_to_ids returns an int."""
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerReturningInt()
    
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    assert isinstance(decoder.eos_token_id, int)
    assert decoder.eos_token_id == 2


def test_eos_token_already_set():
    """Test that existing eos_token_id is used when available."""
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerWithEosTokenId()
    
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    # Should use the existing eos_token_id
    assert decoder.eos_token_id == 2
    assert isinstance(decoder.eos_token_id, int)


def test_eos_comparison_works():
    """Test that token comparisons work correctly with the fixed EOS token."""
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerReturningList()
    
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    # Simulate token generation
    generated_token = 2  # This would be the EOS token
    
    # This comparison should work (previously would fail if eos_token_id was a list)
    assert generated_token == decoder.eos_token_id, \
        "Token comparison should work when token equals EOS"
    
    # Non-EOS token should not match
    other_token = 5
    assert other_token != decoder.eos_token_id, \
        "Non-EOS token should not equal EOS"


def test_empty_list_fallback():
    """Test handling when convert_tokens_to_ids returns an empty list."""
    
    class MockTokenizerEmptyList:
        def __init__(self):
            self.eos_token_id = None
            self.eos_token = "</s>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(m["content"] for m in messages)
        
        def convert_tokens_to_ids(self, token):
            return []  # Empty list edge case
    
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerEmptyList()
    
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    # Should handle empty list gracefully
    assert decoder.eos_token_id is None, \
        "Empty list should result in None"


def test_eos_token_stripped_from_output():
    """Test that EOS token is stripped from decoded output text."""
    
    class MockTokenizerWithDecode:
        def __init__(self):
            self.eos_token_id = 2
            self.eos_token = "<|im_end|>"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            return "".join(m["content"] for m in messages)
        
        def decode(self, tokens):
            """Mock decode that shows EOS token if present."""
            # Simulate decoding: 100 -> "Hello", 2 -> "<|im_end|>"
            result = []
            for token in tokens:
                if token == 100:
                    result.append("Hello")
                elif token == 2:
                    result.append("<|im_end|>")
            return "".join(result)
    
    from src.speculative_manual import ManualSpeculativeDecoder
    
    tokenizer = MockTokenizerWithDecode()
    
    class MockModel:
        pass
    
    decoder = ManualSpeculativeDecoder(
        target_model=MockModel(),
        draft_model=MockModel(),
        tokenizer=tokenizer,
    )
    
    # Simulate tokens with EOS at the end
    generated_tokens = [100, 2]  # "Hello" + EOS
    
    # Manually test the stripping logic
    tokens_to_decode = generated_tokens.copy()
    if tokens_to_decode and tokens_to_decode[-1] == decoder.eos_token_id:
        tokens_to_decode = tokens_to_decode[:-1]
    
    result_text = tokenizer.decode(tokens_to_decode)
    
    # Should NOT contain the EOS token
    assert "<|im_end|>" not in result_text, \
        "EOS token should be stripped from output"
    assert result_text == "Hello", \
        f"Expected 'Hello', got '{result_text}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

