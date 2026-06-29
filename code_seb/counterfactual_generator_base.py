from shared_utils.models.base_model import BaseModel


class CounterfactualProposalResult:
    """An abstract base class for running the pipeline evaluation."""

    def __init__(self, model_name: str, counterfactual_proposal: str, is_valid: bool, proposed_result: str ,*args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.counterfactual_proposal = counterfactual_proposal
        self.is_valid = is_valid
        self.proposed_result = proposed_result

class CounterfactualVerificationResult:
    """A class that collects the results that were calculated in the counterfactual verification phase"""
    def __init__(self, model_name: str, question: str, expected_result: str, is_valid: bool, issue_trace:str = None ,*args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.question = question
        self.expected_result = expected_result
        self.is_valid = is_valid
        self.issue_trace = issue_trace


class CounterfactualGeneratorBase(ABC):
    """A class for generating and verifiying counterfactual generation."""

    def generate_counterfactual_proposal(self, model: BaseModel, base_benchmark_question, expected_result, prompt_template) -> CounterfactualProposalResult:
        """Uses a model to generate a counterfactual proposal.
        Args:
            model (BaseModel): A model to generate a counterfactual proposal.
            base_benchmark_question (str): The base question to modify to generate the counterfactual proposal.
            expected_result (str): The expected result of the counterfactual proposal.
            prompt_template (str, str) -> str: A function that takes in the benchmark question and the expected result and
            produces the prompt to give to the model.

        """
        raise NotImplementedError()

    def verify_counter_factual(self, model: BaseModel, question: str, expected_result: str, prompt_template: str) -> Counterfactual_verification_result:
        """Uses a model to verify a counterfactual proposal."""
        raise NotImplementedError()



