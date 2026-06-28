
class EvaluationRunnerResult:
    """A class that collects the results that were calculated in the evaluation phase"""
    pass


class EvaluationRunnerBase():
    """An abstract base class for running the pipeline evaluation."""

    def run_pipeline(self) -> EvaluationRunnerResult:
        """Run the E2E evaluation pipeline.

        1) Load the dataset that should be tested
        2) Load the model that should be tested
        3) Select the dataset parts for which counterfactuals should be generated
        4) Verify counterfactuals by ensamble of experts and if unsure let human check
        5) See for those wrong if the agent can improve
        6) Return the results


        Returns:

         EvaluationRunnerResult"""

        raise NotImplementedError()

    def load_results(self, output_path: str) -> EvaluationRunnerResult:
        """Load the results from a file.

        Args:
            output_path (str): The path to the output file.

        Returns:
            EvaluationRunnerResult: The loaded results.
        """
        raise NotImplementedError()

    def visualize_results(self, results: EvaluationRunnerResult) -> EvaluationRunnerResult:
        """Visualize the results from a EvaluationRunnerResult.

        Args:
           results (EvaluationRunnerResult): The results to visualize."""
        raise NotImplementedError()

    def write_results(self, results: EvaluationRunnerResult, output_path: str) -> int:
        """Write the results to a file.

        Args:
            results (EvaluationRunnerResult): The results to write.
            output_path (str): The path to the output file.

        Returns:
            int: The number of results written to the file. -1 if unsuccessful.
        """
        raise NotImplementedError()