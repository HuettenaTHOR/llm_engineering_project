

# The primary local ladder for the study. Any other HF repo id also works for ad-hoc testing.
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
    "google/gemma-4-E2B-it",

)

ANTHROPIC_MODELS = ("claude-haiku-4-5-25",)


def load_model_from_str(model_name: str):
    """ loads the model based on the provided model name. """

    if model_name in ANTHROPIC_MODELS:
        from models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    elif model_name in QWEN25_LADDER:
        from models.Qwen25_model import Qwen25Model
        return Qwen25Model(model_name)
    elif model_name in MISTRAL_MODELS:
        from models.mistral_model import MistralModel
        return MistralModel(model_name)
    elif model_name in GEMMA_MODELS:
        from models.gemma_model import GemmaModel
        return GemmaModel(model_name)
    else:
        # Treat anything else as a HuggingFace repo id (Qwen ladder or any model to try out).
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name)