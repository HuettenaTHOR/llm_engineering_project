from shared_utils.models.base_model import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class Qwen25Model(BaseModel):
    def __init__(self, model_name: str, *args, quantize: str = None, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if torch.cuda.is_available():
            # GPU box: shard with accelerate; inputs go to cuda (matches device_map="auto").
            self.device = torch.device("cuda")
            load_kwargs = {"device_map": "auto"}
            if quantize == "8bit":
                # 8-bit lets a 7B fit a 16GB card (bf16 ~15GB overflows it); near-lossless at greedy decode.
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                load_kwargs["dtype"] = "bfloat16"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            # No CUDA (e.g. a Mac): load fully on CPU in fp32. Avoids the device_map="auto"
            # MPS placement mismatch and MPS op-coverage gaps; fine for small smoke runs.
            self.device = torch.device("cpu")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float32
            ).to(self.device)
        print(f"Using device: {self.device}")

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