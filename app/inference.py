import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src import model as M
from src import utils

# Global variables for model and utilities
_model = None
_itos = None
_device = None


def load_model():
    """Load the model and utilities once with caching.

    Returns:
        tuple: (model, itos, device) where model is the loaded BigramLanguageModel,
               itos is the index-to-string mapping, and device is the PyTorch device
    """
    global _model, _itos, _device

    if _model is None:
        text, chars, stoi, itos, encode, decode = utils.load_data()
        vocab_size = len(chars)
        _device = config.get_device()
        _model = M.BigramLanguageModel(vocab_size)
        _model.load_state_dict(
            torch.load("checkpoints/tiny_shakespeare.pt", map_location=_device)
        )
        _model.to(_device)
        _model.eval()
        _itos = itos

    return _model, _itos, _device


def generate_text_stream(max_new_tokens=1000, seed_text=""):
    """Generate text from the model with streaming support

    Args:
        max_new_tokens (int): Number of tokens to generate
        seed_text (str): Optional seed text to start generation

    Yields:
        str: Generated characters one by one
    """
    model, itos, device = load_model()

    if seed_text:
        text, chars, stoi, itos_temp, encode, decode = utils.load_data()
        try:
            context = torch.tensor([encode(seed_text)], dtype=torch.long, device=device)
        except KeyError:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = context[:, -config.BLOCK_SIZE :]
            logits, loss = model(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            token_id = idx_next[0, 0].item()
            char = itos[token_id]
            yield char

            context = torch.cat((context, idx_next), dim=1)


if __name__ == "__main__":
    # For printing text character by character
    import time

    print("Generated text:\n")
    for char in generate_text_stream(max_new_tokens=500):
        print(char, end="", flush=True)
        time.sleep(0.015)
    print("\n")
