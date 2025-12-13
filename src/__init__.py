# Speculative Decoding with Adaptive LoRA Training
# MLX implementation for Apple Silicon

__version__ = "0.1.0"

# Data collection (no MLX dependency)
from .data_collector import (
    TokenLevelDisagreement,
    TrainingExample,
    DataCollector,
    AcceptanceRateTracker,
)

# MLX-dependent modules - import conditionally
# These require MLX which is only available on Apple Silicon
try:
    from .models import (
        ModelManager,
        sample_token,
        get_logits,
        create_kv_cache,
        get_logits_with_cache,
        rewind_cache,
        get_cache_length,
    )
    from .speculative import (
        GenerationMetrics,
        SpeculativeResult,
        SpeculativeDecoder,
        run_acceptance_benchmark,
    )
    from .speculative_manual import (
        ManualSpeculativeMetrics,
        ManualSpeculativeResult,
        ManualSpeculativeDecoder,
        run_data_collection_batch,
    )

    _MLX_AVAILABLE = True
except ImportError:
    # MLX not available (likely not on Apple Silicon)
    _MLX_AVAILABLE = False
    ModelManager = None
    sample_token = None
    get_logits = None
    create_kv_cache = None
    get_logits_with_cache = None
    rewind_cache = None
    get_cache_length = None
    GenerationMetrics = None
    SpeculativeResult = None
    SpeculativeDecoder = None
    run_acceptance_benchmark = None
    ManualSpeculativeMetrics = None
    ManualSpeculativeResult = None
    ManualSpeculativeDecoder = None
    run_data_collection_batch = None

__all__ = [
    # Data collection (always available)
    "TokenLevelDisagreement",
    "TrainingExample",
    "DataCollector",
    "AcceptanceRateTracker",
    # Model management (requires MLX)
    "ModelManager",
    "sample_token",
    "get_logits",
    "create_kv_cache",
    "get_logits_with_cache",
    "rewind_cache",
    "get_cache_length",
    # Fast speculative decoding (requires MLX)
    "GenerationMetrics",
    "SpeculativeResult",
    "SpeculativeDecoder",
    "run_acceptance_benchmark",
    # Manual speculative decoding (requires MLX)
    "ManualSpeculativeMetrics",
    "ManualSpeculativeResult",
    "ManualSpeculativeDecoder",
    "run_data_collection_batch",
]
