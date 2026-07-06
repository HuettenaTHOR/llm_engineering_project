

# The primary local ladder for the study. Any other HF repo id also works for ad-hoc testing.
# Suggested ladder (the real Qwen3.5 small series). NOT pinned -- any HF repo id passed as
# --model also works (it falls through to HuggingFaceModel below), so newer models are testable.


QWEN35_LADDER = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
)

QWEN25_LADDER = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

MISTRAL_MODELS = (
    "mistralai/Ministral-3-3B-Instruct-2512",
    "mistralai/Ministral-3-8B-Instruct-2512"
)

GEMMA_MODELS = (
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",

)

# Standard-chat causal LMs (no <think> block); loaded via the generic HuggingFaceModel.
LLAMA_LADDER = (
    "meta-llama/Llama-3.1-8B-Instruct",  # gated: request access + `huggingface-cli login` first
)

PHI_MODELS = (
    "microsoft/Phi-4-mini-instruct",  # 3.8B, fits bf16
    "microsoft/phi-4",                # 14B, needs 8-bit on a 16GB card
)

ANTHROPIC_MODELS = ("claude-haiku-4-5",)

# Repos whose bf16 footprint overflows a 16GB card, so they default to 8-bit (near-lossless at
# greedy decode). Ministral is excluded -- MistralModel loads its own native FP8 checkpoint.
_NEEDS_8BIT = frozenset({
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "microsoft/phi-4",
    "google/gemma-4-E4B-it",
})


def load_model_from_str(model_name: str, quantize: str = None):
    """Loads the model for the given repo id. ``quantize`` (e.g. "8bit") forces low-bit loading;
    when left None, models in ``_NEEDS_8BIT`` auto-select 8-bit so they fit the 16GB card."""
    if quantize is None and model_name in _NEEDS_8BIT:
        quantize = "8bit"

    if model_name in ANTHROPIC_MODELS:
        from shared_utils.models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    elif model_name in QWEN35_LADDER:
        # Qwen3.5 is a thinking model (<think> block); its own class handles enable_thinking.
        from shared_utils.models.qwen35_model import Qwen35Model
        return Qwen35Model(model_name)
    elif model_name in QWEN25_LADDER:
        from shared_utils.models.qwen25_model import Qwen25Model
        return Qwen25Model(model_name, quantize=quantize)
    elif model_name in MISTRAL_MODELS:
        from shared_utils.models.mistral_model import MistralModel
        return MistralModel(model_name)
    elif model_name in GEMMA_MODELS:
        from shared_utils.models.gemma_model import GemmaModel
        return GemmaModel(model_name, quantize=quantize)
    else:
        # Llama, Phi, or any other standard-chat HF repo id (also the fallback for ad-hoc testing).
        from shared_utils.models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name, quantize=quantize)