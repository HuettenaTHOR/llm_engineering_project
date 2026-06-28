from models.base_model import BaseModel
import anthropic

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from the env

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float = 0.0):
        """
        This method implements the inference logic for the Anthropic model."""
        # Anthropic takes the system prompt separately, not as a role in the messages list.
        system, messages = self.split_system(conversation)
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return next(block.text for block in response.content if block.type == "text")

    def split_system(self, conversation: list):
        """Lift the system message out of the conversation and return (system, messages)."""
        system = ""
        messages = []
        for message in conversation:
            if message is None:
                continue
            if message["role"] == "system":
                system = message["content"]
            else:
                messages.append({"role": message["role"], "content": message["content"]})
        return system, messages
