from abc import ABC

class BaseModel(ABC):
    """
    Base class for all models we want to test. This class is abstract and should not be instantiated directly.
    """

    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float | None = None,
                  max_time: float | None = 300) -> str:
        """
        This method should include the inference logic for the model. temperature=None -> use the
        model's shipped generation_config defaults; an explicit float overrides (0.0 -> greedy).
        max_time is a per-generation wall-clock cap (seconds) to bound runaway generations; local
        HF models pass it to generate(). API models (e.g. Anthropic) may ignore it."""
        raise NotImplementedError("The inference method must be implemented in the subclass.")
    
    def build_conversation(self, conversation: list):
        """
        This method should include the logic to build a conversation from the input data. """
        raise NotImplementedError("The build_conversation method must be implemented in the subclass.")
    
    def build_conversation_from_system_prompt(self, system_prompt: str, user_input: str = None):
        messages = [{"role": "system", "content": system_prompt}]
        if user_input is not None:
            messages.append({"role": "user", "content": user_input})
        return messages
