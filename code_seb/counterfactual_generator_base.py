from shared_utils.models.base_model import BaseModel


class CounterfactualProposalResult:
    """An abstract base class for running the pipeline evaluation."""

    def __init__(self, model_name: str, counterfactual_proposal: str, is_valid: bool, proposed_result: str ,*args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.counterfactual_proposal = counterfactual_proposal
        self.is_valid = is_valid
        self.proposed_result = proposed_result


class CounterfactualGenerator(ABC):
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

