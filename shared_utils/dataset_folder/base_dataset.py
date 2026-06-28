from abc import ABC

class BaseModel(ABC):

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset = self.load_dataset()

    def load_dataset(self):
        """
        This method should include the logic to load the dataset. """
        raise NotImplementedError("The load_dataset method must be implemented in the subclass.")
    
    def get_single_example(self, index: int):
        """
        This method should include the logic to get a single example from the dataset. """
        raise NotImplementedError("The get_single_example method must be implemented in the subclass.")
    
    def get_dataset_size(self):
        """
        This method should include the logic to get the size of the dataset. """
        raise NotImplementedError("The get_dataset_size method must be implemented in the subclass.")
    
    def get_dataset(self):
        """
        This method should include the logic to get the entire dataset. """
        raise NotImplementedError("The get_dataset method must be implemented in the subclass.")
    
    def get_random_subset(self, size: int, seed: int = 42):
        """
        This method should include the logic to get a random subset of the dataset. """
        raise NotImplementedError("The get_random_subset method must be implemented in the subclass.")