def load_dataset_of_string(dataset_name: str):
    """
    Load a dataset based on the provided dataset name.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset.
    """
    if dataset_name == "gsm8k":
        from dataset_folder.gsm8k_dataset import GSM8KDataset
        return GSM8KDataset()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")