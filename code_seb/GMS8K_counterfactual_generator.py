from code_seb.counterfactual_generator import CounterfactualGenerator
from shared_utils.dataset_folder.gsm8k_dataset import GSM8KDataset
from shared_utils.models.base_model import BaseModel


class GMS8KCounterfactualGenerator(CounterfactualGenerator):

    def __init__(self, test_model: BaseModel, verifier_model: BaseModel, **kwargs):
        super().__init__(**kwargs)

        # initialize the dataset
        self.gms8k = GSM8KDataset()
        self.gms8k.load_dataset()

        self.verifier_model = verifier_model # the model to verify the counterfactuals
        self.test_model = test_model # the model to be tested




    def prompt_template(self, base_benchmark_question: str, expected_result: str) -> str:
        """
        This method generates a prompt for the model to generate a counterfactual proposal.
        Args:
            base_benchmark_question (str): The base question to modify to generate the counterfactual proposal.
            expected_result (str): The expected result of the counterfactual proposal.
        Returns:
            str: The prompt to give to the model.
        """

        prompt = f"""
        You are a math professor that generates a new questions for the GMS8K benchmark.
        Given the following base question and expected result, generate a new question that is similar in structure but has a different expected result.
        
        Base Question: {base_benchmark_question}
        Expected Result: {expected_result}
        
        Please provide the new question below:
        """
        return prompt

    def extract_is_valid(self, inference_result: str, expected_result: str) -> bool:
        """
        This method extracts whether the model's inference result is valid based on the expected result.
        Args:
            inference_result (str): The result from the model's inference.
            expected_result (str): The expected result of the counterfactual proposal.
        Returns:
            bool: True if the inference result is valid, False otherwise.

        """
        extracted_result = self.gms8k.postprocess_result(inference_result)
        return extracted_result == expected_result





