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
        # Extract the gold answer string after the last '####' marker, or None.
        if result is not None and "####" in result:
            return result.split("####")[-1].strip()
        return None
   
    def system_prompt(self, **kwargs) -> str:
        return """You will be given a math problem. The solution to the problem is an integer. Your task is to provide the solution. Only provide the final answer as an integer. Think step by step. Always end your answer with a number. The number should be given after '####'. e.g. if the answer is 78, answer with #### 78. The math problem is: {PROBELM}""".format(PROBELM=kwargs.get("problem", ""))