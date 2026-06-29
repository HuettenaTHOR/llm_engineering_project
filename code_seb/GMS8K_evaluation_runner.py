import json

from code_seb.evaluation_runner_base import EvaluationRunnerBase, EvaluationRunnerResult
from code_seb.counterfactual_generator_base import CounterfactualProposalResult, CounterfactualVerificationResult
from code_seb.GMS8K_counterfactual_generator import GMS8KCounterfactualGenerator
from shared_utils.models.base_model import BaseModel


class GMS8KEvaluationRunnerResult(EvaluationRunnerResult):
    def __init__(self, results: list[tuple[CounterfactualProposalResult, CounterfactualVerificationResult]]):
        self.results = results


class GMS8KEvaluationRunner(EvaluationRunnerBase):

    def __init__(self, test_model: BaseModel, verifier_model: BaseModel, n: int):
        self.generator = GMS8KCounterfactualGenerator(test_model=test_model, verifier_model=verifier_model)
        self.n = n

    def run_pipeline(self) -> GMS8KEvaluationRunnerResult:
        results = self.generator.generate_and_verify_N(self.n)
        return GMS8KEvaluationRunnerResult(results)

    def load_results(self, output_path: str) -> GMS8KEvaluationRunnerResult:
        with open(output_path, "r") as f:
            raw = json.load(f)
        results = []
        for entry in raw:
            proposal = object.__new__(CounterfactualProposalResult)
            proposal.__dict__.update(entry["proposal"])
            verification = object.__new__(CounterfactualVerificationResult)
            verification.__dict__.update(entry["verification"])
            results.append((proposal, verification))
        return GMS8KEvaluationRunnerResult(results)

    def visualize_results(self, results: GMS8KEvaluationRunnerResult):
        total = len(results.results)
        valid = sum(1 for _, v in results.results if v.is_valid)
        print(f"Results: {valid}/{total} counterfactuals verified valid")
        print("-" * 60)
        for i, (proposal, verification) in enumerate(results.results, 1):
            status = "VALID" if verification.is_valid else "INVALID"
            print(f"[{i}] {status} | expected: {verification.expected_result}")
            print(f"     {proposal.counterfactual_proposal[:120].strip()}...")
            if verification.issue_trace:
                print(f"     Issue: {verification.issue_trace}")

    def write_results(self, results: GMS8KEvaluationRunnerResult, output_path: str) -> int:
        try:
            serialized = [
                {"proposal": vars(proposal), "verification": vars(verification)}
                for proposal, verification in results.results
            ]
            with open(output_path, "w") as f:
                json.dump(serialized, f, indent=2)
            return len(serialized)
        except IOError:
            return -1


if __name__ == "__main__":
    from shared_utils.models import load_model_from_str, ANTHROPIC_MODELS, GEMMA_MODELS

    test_model = load_model_from_str(GEMMA_MODELS[0])
    verifier_model = load_model_from_str(ANTHROPIC_MODELS[0])

    runner = GMS8KEvaluationRunner(test_model, verifier_model, n=5)
    results = runner.run_pipeline()
    runner.visualize_results(results)
    runner.write_results(results, "results.json")
