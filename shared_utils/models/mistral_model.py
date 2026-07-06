from shared_utils.models.base_model import BaseModel
from transformers import Mistral3ForConditionalGeneration, FineGrainedFP8Config, AutoTokenizer
import torch

class MistralModel(BaseModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        quantization_config = FineGrainedFP8Config(
            dequantize=True
        )
        self.model = Mistral3ForConditionalGeneration.from_pretrained(model_name, device_map="auto", quantization_config=quantization_config)

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float | None = None) -> str:
        """
        This method implements the inference logic for the HuggingFace model."""
        input_text = self.build_conversation(conversation)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        gen_kwargs = {"max_new_tokens": max_tokens}
        if temperature is not None:  # explicit float -> temp 0 greedy (reproducible), else sample
            gen_kwargs["do_sample"] = temperature > 0
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
        # temperature is None -> no overrides; generation_config.json decides do_sample/temp/top_p/top_k
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)



    def build_conversation(self, conversation: list):
        """
        This method implements the logic to build a conversation from the input data for the HuggingFace model."""
        # assume the conversation is already in the correct format for the HuggingFace model
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
