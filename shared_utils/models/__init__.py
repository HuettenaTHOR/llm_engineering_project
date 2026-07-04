

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

ANTHROPIC_MODELS = ("claude-haiku-4-5",)


def load_model_from_str(model_name: str, quantize: str = None):
    """ loads the model based on the provided model name. ``quantize`` (e.g. "8bit") lets a large
    model (7B) fit a 16GB card; currently honored by the Qwen2.5 loader. """

    if model_name in ANTHROPIC_MODELS:
        from shared_utils.models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    elif model_name in QWEN35_LADDER:
        # Qwen3.5 instruct models use the standard system/user chat shape.
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
        # Treat anything else as a HuggingFace repo id (Qwen ladder or any model to try out).
        from shared_utils.models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name)