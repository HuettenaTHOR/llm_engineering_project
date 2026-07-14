from shared_utils.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen35Model(BaseModel):
    # Qwen3.5 is a thinking model. Set this False on an instance (e.g. a verifier) to suppress the
    # <think> block -> concise, fast, and no truncating-before-the-verdict on hard problems.
    enable_thinking = True

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        try:
            # trust_remote_code + dtype="auto" keep newer architectures (e.g. Qwen3.5)
            # loadable without hardcoding a dtype the checkpoint may not ship.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", dtype="auto", trust_remote_code=True,
            )
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Could not load model '{model_name}'.\n"
                f"Original error: {e}"
            ) from e

    def inference(self, conversation: list, max_tokens: int = 2500, temperature: float | None = None,
                  max_time: float | None = 300):
        """
        This method implements the inference logic for the Qwen35 model."""
        input_text = self.build_conversation(conversation)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        gen_kwargs = {"max_new_tokens": max_tokens}
        if max_time is not None:
            # Wall-clock cap per generation (seconds): bounds runaway generations that ramble to
            # max_tokens near the VRAM cap (observed multi-hour items). transformers stops at
            # max_time and returns the text so far; fast/normal generations finish well under it.
            gen_kwargs["max_time"] = max_time
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
        This method implements the logic to build a conversation from the input data for the Qwen35 model."""
        # assume the conversation is already in the correct format for the Qwen35 model
        return self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        