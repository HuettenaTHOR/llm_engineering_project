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

LLAMA_LADDER = (
    "meta-llama/Llama-3.1-8B-Instruct",  
)

ANTHROPIC_MODELS = ("claude-haiku-4-5",)

OPENROUTER_MODELS = (
    "mistralai/mistral-small-2603",
    "google/gemini-3.1-flash-lite",
    "openai/gpt-5.4-nano",
    "openai/gpt-5.6-luna",
    "deepseek/deepseek-v4-flash",
)

_NEEDS_8BIT = frozenset({
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
})


def load_model_from_str(model_name: str, quantize: str = None):
    """Loads the model for the given repo id. ``quantize`` (e.g. "8bit") forces low-bit loading;
    when left None, models in ``_NEEDS_8BIT`` auto-select 8-bit so they fit the 16GB card."""
    if quantize is None and model_name in _NEEDS_8BIT:
        quantize = "8bit"
    if model_name in ANTHROPIC_MODELS:
        from shared_utils.models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    elif model_name in OPENROUTER_MODELS:
        from shared_utils.models.openrouter_model import OpenRouterModel
        return OpenRouterModel(model_name)
    elif model_name in QWEN35_LADDER:
        # Qwen3.5 is a thinking model (<think> block); its own class handles enable_thinking.
        from shared_utils.models.qwen35_model import Qwen35Model
        return Qwen35Model(model_name)
    elif model_name in QWEN25_LADDER:
        from shared_utils.models.qwen25_model import Qwen25Model
        return Qwen25Model(model_name, quantize=quantize)
    else:
        # Llama, Phi, or any other standard-chat HF repo id (also the fallback for ad-hoc testing).
        from shared_utils.models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name, quantize=quantize)