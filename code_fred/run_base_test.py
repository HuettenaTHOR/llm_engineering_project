from fixed_seeds import set_all_seeds
from dataset_folder import load_dataset_of_string
from models import load_model_from_str
from tqdm import tqdm
from answer_extraction import extract_float


def run_test_reasoning_baseline(model_name, dataset_name):
    set_all_seeds(42)
    model = load_model_from_str(model_name)
    dataset = load_dataset_of_string(dataset_name)

    metrics = {"correct": 0, "total": 0, "gen_fail": 0}

    subset = dataset.get_random_subset(size=10, seed=42)  # small smoke subset for testing
    for i in tqdm(range(len(subset)), desc=f"Testing {model_name} on {dataset_name}"):
        example = subset[i]
        input_text = example["question"]
        expected_output = example["answer"]
        sys_prompt = dataset.system_prompt(problem=input_text)
        # Get the model's output
        input_conversation = model.build_conversation_from_system_prompt(sys_prompt, user_input=input_text)
        model_output = model.inference(input_conversation, max_tokens=1200, temperature=1.0)

        # Grade on extracted integers, not full-string equality (Issue #1 fix).
        pred = extract_float(model_output)
        gold = extract_float(dataset.postprocess_result(expected_output))
        print(f"\n\nExpected: {gold}\nModel pred: {pred}\nRaw output: {model_output}\n")

        if pred is None:
            metrics["gen_fail"] += 1  # unparseable output: not scored as correct
        elif gold is not None and int(pred) == int(gold):
            metrics["correct"] += 1
        metrics["total"] += 1

    accuracy = metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0
    print(
        f"Model: {model_name}, Dataset: {dataset_name}, "
        f"Accuracy: {accuracy:.2%} ({metrics['correct']}/{metrics['total']}), "
        f"Gen-fail: {metrics['gen_fail']}/{metrics['total']}"
    )


if __name__ == "__main__":
    # Example usage
    run_test_reasoning_baseline("google/gemma-4-E2B-it", "gsm8k")
