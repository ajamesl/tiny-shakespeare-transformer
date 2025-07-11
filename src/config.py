import torch

_device = None  # Internal cache to prevent repeated device checks/prints

def get_device():
    """Determine and return the best available device for PyTorch computations."""
    global _device
    if _device is not None:
        return _device

    if torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"Using device: {_device}")
    return _device


# Training hyperparameters
BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 1000
LEARNING_RATE = 3e-4
EVAL_ITERS = 200

# Model hyperparameters
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
