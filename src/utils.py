def load_data():
    """Load the Shakespeare dataset and create character mappings."""
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        """Encode string to list of integers."""
        return [stoi[c] for c in s]

    def decode(tokens):
        """Decode list of integers to string."""
        return "".join([itos[i] for i in tokens])

    return text, chars, stoi, itos, encode, decode


def split_data(data_tensor):
    """Split data tensor into training and validation sets.

    Args:
        data_tensor: Input tensor to be split

    Returns:
        tuple: (train_data, val_data) where train_data contains 90% of the data
               and val_data contains the remaining 10%
    """
    n = int(0.9 * len(data_tensor))
    return data_tensor[:n], data_tensor[n:]
