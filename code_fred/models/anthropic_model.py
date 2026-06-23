from models.base_model import BaseModel
import anthropic

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.client = anthropic.Client(api_key=kwargs.get("api_key"))

    def inference(self, conversation: list, max_tokens: int = 1000):
        """
        This method implements the inference logic for the Anthropic model."""
        input_text = self.build_conversation(conversation)
        response = self.client.completions.create(
            model=self.model_name,
            prompt=input_text,
            max_tokens_to_sample=max_tokens
        )
        return response.completion
    
    def build_conversation(self, conversation: list):
        """
        This method implements the logic to build a conversation from the input data for the Anthropic model."""
        # assume the conversation is in the transformers format and we need to convert it to a string for the Anthropic model
        output_conversation = ""
        for message in conversation:
            role = message.get("role")
            content = message.get("content")
            if role and content:
                output_conversation += f"{role}: {content}\n"
        return output_conversation
