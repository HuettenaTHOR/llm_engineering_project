import hashlib
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


def set_item_seed(seed: int, item_id: str):
    """Deterministic per-item RNG state so item i starts identically in every condition and on
    resume (single-shot vs loop consume the RNG differently, which desyncs a single run-level seed)."""
    h = int(hashlib.sha1(f"{seed}:{item_id}".encode()).hexdigest()[:15], 16)
    random.seed(h)
    np.random.seed(h % (2**32))
    torch.manual_seed(h)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(h)
    transformers.set_seed(h % (2**32))
