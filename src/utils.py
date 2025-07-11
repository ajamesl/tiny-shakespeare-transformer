"""Utility functions for data loading and preprocessing."""

from typing import Callable

__all__ = ["load_data", "split_data"]


def load_data() -> tuple[
    str,
    list[str],
    dict[str, int],
    dict[int, str],
    Callable[[str], list[int]],
    Callable[[list[int]], str],
]:
    """Load the Shakespeare dataset and create character mappings."""
    try:
        with open("data/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError("Shakespeare dataset not found at data/input.txt")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        """Encode string to list of integers."""
        return [stoi[c] for c in s]

    def decode(tokens: list[int]) -> str:
        """Decode list of integers to string."""
        return "".join([itos[i] for i in tokens])

    return text, chars, stoi, itos, encode, decode


def split_data(data_tensor) -> tuple:
    """Split data tensor into training and validation sets.

    Args:
        data_tensor: Input tensor to be split

    Returns:
        tuple: (train_data, val_data) where train_data contains 90% of the data
               and val_data contains the remaining 10%
    """
    n = int(0.9 * len(data_tensor))
    return data_tensor[:n], data_tensor[n:]
