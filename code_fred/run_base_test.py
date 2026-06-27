import re

from dataset_folder import load_dataset_of_string
from models import load_model_from_str
from tqdm import tqdm

from CONSTANTS import SYSTEM_PROMPT


# TODO(Issue #2): replace this stopgap with code_fred/extraction.py::extract_int
# (3-tier: last '####' regex -> last-integer fallback -> Gen-fail). Kept minimal
# here so Issue #1 grading is correct without pulling in #2 early.
def _extract_last_int(text):
    """Return the last integer found in `text` ($/commas stripped), or None."""
    if text is None:
        return None
    matches = re.findall(r"-?\d[\d,]*", str(text))
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def run_test_reasoning_baseline(model_name, dataset_name):
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
        input_conversation = model.build_conversation_from_system_prompt(sys_prompt)
        model_output = model.inference(input_conversation, max_tokens=600)

        # Grade on extracted integers, not full-string equality (Issue #1 fix).
        pred = _extract_last_int(model_output)
        gold = _extract_last_int(dataset.postprocess_result(expected_output))
        print(f"\n\nExpected: {gold}\nModel pred: {pred}\nRaw output: {model_output}\n")

        if pred is None:
            metrics["gen_fail"] += 1  # unparseable output: not scored as correct
        elif gold is not None and pred == gold:
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
    run_test_reasoning_baseline("Qwen/Qwen2.5-7B-Instruct", "gsm8k")
