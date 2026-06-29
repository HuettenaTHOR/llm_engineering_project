from code_seb.counterfactual_generator_base import CounterfactualGeneratorBase, CounterfactualProposalResult, CounterfactualVerificationResult
from shared_utils.models.base_model import BaseModel


class CounterfactualGenerator(CounterfactualGeneratorBase):
    """Implements functions to generate and verify counterfactuals."""

    @overrides
    def generate_counterfactual(self, model: BaseModel, base_benchmark_question, expected_result, prompt_template) -> CounterfactualProposalResult:
        """Uses a model to generate a counterfactual proposal.
        Args:
            model (BaseModel): A model to generate a counterfactual proposal.
            base_benchmark_question (str): The base question to modify to generate the counterfactual proposal.
            expected_result (str): The expected result of the counterfactual proposal.
            prompt_template (str, str) -> str: A function that takes in the benchmark question and the expected result and
            produces the prompt to give to the model.

        """
        # some assertions to make the method robust
        assert(isinstance(model, BaseModel))
        assert(base_benchmark_question is not None)
        assert(expected_result is not None)
        assert (prompt_template is not None)


        prompt = prompt_template(base_benchmark_question, expected_result)
        inference_result = model.inference(prompt)

        return CounterfactualProposalResult(
            model.model_name,
            counterfactual_proposal=inference_result,
            is_valid=None,  # This is a placeholder; actual validation logic should be implemented.
            proposed_result=expected_result,
            prompt_template=prompt,
        )



    @overrides
    def verify_counter_factual(self, model: BaseModel, question: str, expected_result: str, prompt_template: str, extract_is_valid: bool) -> Counterfactual_verification_result:
        """Uses a model to verify a counterfactual proposal."""
        assert(isinstance(model, BaseModel))
        assert(question is not None)
        assert(expected_result is not None)
        assert(prompt_template is not None)

        prompt = prompt_template(question, expected_result)
        inference_result = model.inference(prompt)

        is_valid = extract_is_valid(inference_result, expected_result)

        return CounterfactualVerificationResult(
            model.model_name,
            counterfactual_proposal=question,
            is_valid=is_valid,
            expected_result=expected_result,
        )