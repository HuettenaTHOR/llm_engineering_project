import random
import numpy as np
# Import pyarrow before torch: on Windows, loading torch first makes pyarrow's native
# libs crash with an access violation (0xC0000005) when datasets imports them later.
import pyarrow.dataset  # noqa: F401  (ordering side effect only)
import torch
import transformers


def set_all_seeds(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)
