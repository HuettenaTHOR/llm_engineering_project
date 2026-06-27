

def load_model_from_str(model_name: str):
    """ loads the model based on the provided model name. """

    if model_name == "mistralai/Ministral-3-3B-Instruct-2512":
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name)
    elif model_name == "Qwen/Qwen3.5-2B":
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name)
    elif model_name in ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"):
        from models.huggingface_model import HuggingFaceModel
        return HuggingFaceModel(model_name)
    elif model_name == "claude-haiku-4-5-20251001":
        from models.anthropic_model import AnthropicModel
        return AnthropicModel(model_name)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")