from dataset_folder.base_dataset import BaseModel
from datasets import load_dataset

class GSM8KDataset(BaseModel):
    def __init__(self):
        super().__init__(dataset_name="gsm8k")

    def load_dataset(self):
        return load_dataset("openai/gsm8k", "main", split="test")
    
    def get_single_example(self, index: int):
        return self.dataset[index]
    
    def get_dataset_size(self):
        return len(self.dataset)
    
    def get_dataset(self):
        return self.dataset
    
    def get_random_subset(self, size: int, seed: int = 42):
        return self.dataset.shuffle(seed=seed).select(range(size))
    
    def postprocess_result(self, result: str):
        # Postprocess the result if needed (e.g., strip whitespace)
        if "####" in result:
            return result.split("####")[1].strip()
        else:
            None
        
    def prompt_addition_for_output_tracing(self) -> str:
        return "Always end your answer with a number. The number should be given after '####'. e.g. if the answer is 78, answer withh #### 78."