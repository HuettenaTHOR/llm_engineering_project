from dataset_folder import load_dataset_of_string
from models import load_model_from_str
from tqdm import tqdm

from CONSTANTS import SYSTEM_PROMPT

def run_test_reasoning_baseline(model_name, dataset_name):
    model = load_model_from_str(model_name)
    dataset = load_dataset_of_string(dataset_name)

    metrics = {"correct": 0, "total": 0}
    
    subset = dataset.get_random_subset(size=10, seed=42)  # Get a random subset of 10 examples for testing
    for i in tqdm(range(len(subset)), desc=f"Testing {model_name} on {dataset_name}"):
        example = subset[i]
        input_text = example['question']
        expected_output = example['answer']
        sys_prompt = SYSTEM_PROMPT + dataset.prompt_addition_for_output_tracing
        # Get the model's output
        input_conversation = model.build_conversation_from_system_prompt(sys_prompt, input_text)
        model_output = model.inference(input_conversation)
        print(f"Input: {input_text}\nExpected Output: {expected_output}\nModel Output: {model_output}\n")
        # Compare the model's output with the expected output
        if model_output.strip() == expected_output.strip():
            metrics["correct"] += 1
        metrics["total"] += 1

    accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    print(f"Model: {model_name}, Dataset: {dataset_name}, Accuracy: {accuracy:.2%} ({metrics['correct']}/{metrics['total']})")

if __name__ == "__main__":
    # Example usage
    run_test_reasoning_baseline("Qwen/Qwen2.5-0.5B", "gsm8k")
    

