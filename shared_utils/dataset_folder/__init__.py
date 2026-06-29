def load_dataset_of_string(dataset_name: str):
    """
    Load a dataset based on the provided dataset name.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset.
    """
    print(f"Loading dataset '{dataset_name}'...")
    if dataset_name == "gsm8k":
        from shared_utils.dataset_folder.gsm8k_dataset import GSM8KDataset
        print("import done")
        return GSM8KDataset()
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")