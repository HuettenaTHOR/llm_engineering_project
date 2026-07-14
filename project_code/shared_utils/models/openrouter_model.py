from shared_utils.models.base_model import BaseModel
from openai import OpenAI
import os

class OpenRouterModel(BaseModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float | None = None) -> str:
        """
        This method implements the inference logic for OpenRouter-hosted models."""
        # OpenRouter's chat/completions endpoint is OpenAI-compatible and accepts the
        # conversation list (including a "system" role message) as-is.
        temp_kwargs = {} if temperature is None else {"temperature": temperature}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=conversation,
            max_tokens=max_tokens,
            **temp_kwargs,
        )
        return response.choices[0].message.content
