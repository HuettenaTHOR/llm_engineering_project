from shared_utils.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HuggingFaceModel(BaseModel):
    """Generic standard-chat causal LM (system/user/assistant, no <think> block). Handles any HF
    repo id that follows the ordinary chat-template convention -- e.g. Llama 3.1 and Phi-4."""

    def __init__(self, model_name: str, *args, quantize: str = None, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        try:
            # trust_remote_code keeps newer architectures loadable without hardcoding config.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            load_kwargs = {"device_map": "auto", "trust_remote_code": True}
            if quantize == "8bit":
                # 8-bit lets an ~8B/14B model fit a 16GB card (bf16 overflows it); near-lossless
                # at greedy decode. bitsandbytes owns the dtype, so don't also pass dtype="auto".
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                # dtype="auto" respects whatever the checkpoint ships (bf16/fp16).
                load_kwargs["dtype"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"Could not load model '{model_name}'.\n"
                f"  - Check the repo id is correct and public on HuggingFace.\n"
                f"  - For brand-new architectures, update transformers: "
                f"`pip install -U transformers`.\n"
                f"  - For gated repos, log in first: `huggingface-cli login`.\n"
                f"Original error: {e}"
            ) from e

    def inference(self, conversation: list, max_tokens: int = 1000, temperature: float | None = None,
                  max_time: float | None = 300) -> str:
        """
        This method implements the inference logic for the HuggingFace model."""
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
        This method implements the logic to build a conversation from the input data for the HuggingFace model."""
        # assume the conversation is already in the correct format for the HuggingFace model
        return self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        