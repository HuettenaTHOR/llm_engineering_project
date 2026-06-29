from tasks.base_task import BaseTask, to_int
from answer_extraction import extract_float


class SolveTask(BaseTask):
    """Plain GSM8K solving: answer the question, graded against the gold ``####`` integer."""

    def __init__(self, dataset):
        self.dataset = dataset

    def build_messages(self, example, model) -> list:
        question = example["question"]
        system = self.dataset.system_prompt(problem=question)
        return model.build_conversation_from_system_prompt(system, user_input=question)

    # build_verifier_messages is inherited from BaseTask: a static step-checking verifier that
    # reviews the solver's full output and answers yes/no (no dataset-specific prompt).

    def gold(self, example):
        """The gold integer parsed from the dataset answer's ``####`` marker, or None."""
        return to_int(extract_float(self.dataset.postprocess_result(example["answer"])))

    def grade(self, example, model_output: str) -> dict:
        pred = self.parse_answer(model_output)
        gold = self.gold(example)
        if pred is None:
            return {"correct": None, "gen_fail": True, "pred": None}
        return {"correct": gold is not None and pred == gold, "gen_fail": False, "pred": pred}
