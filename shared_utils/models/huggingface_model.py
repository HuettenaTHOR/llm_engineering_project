from models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype="bfloat16")

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float = 0.0):
        """
        This method implements the inference logic for the HuggingFace model."""
        input_text = self.build_conversation(conversation)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        do_sample = temperature > 0  # temp 0 -> greedy, so runs are reproducible
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)



    def build_conversation(self, conversation: list):
        """
        This method implements the logic to build a conversation from the input data for the HuggingFace model."""
        # assume the conversation is already in the correct format for the HuggingFace model
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        