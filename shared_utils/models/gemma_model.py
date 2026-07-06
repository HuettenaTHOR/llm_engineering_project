from shared_utils.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class GemmaModel(BaseModel):
    # Gemma is a thinking model. Set this False on an instance (e.g. a verifier) to suppress the
    # <think> block -> concise, fast, and no truncating-before-the-verdict on hard problems.
    enable_thinking = True

    def __init__(self, model_name: str, *args, quantize: str = None, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 8-bit halves the footprint so the ~16GB bf16 model fits a 16GB card without CPU offload;
        # near-lossless at greedy decode. (The full bf16 shard is still mmapped during load, so an
        # adequate Windows page file is required regardless -- quantization shrinks the resident model,
        # not the on-disk load.)
        load_kwargs = {"device_map": "auto"}
        if quantize == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["dtype"] = "bfloat16"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

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
                gen_kwargs["top_p"] = 0.95  # based on transformers docs
                gen_kwargs["top_k"] = 64    # based on transformers docs
        # temperature is None -> no overrides; generation_config.json decides do_sample/temp/top_p/top_k

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)



    def build_conversation(self, conversation: list):
        """
        This method implements the logic to build a conversation from the input data for the HuggingFace model."""
        # assume the conversation is already in the correct format for the HuggingFace model
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, enable_thinking=self.enable_thinking)
