def load_data(path='data/input.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return text, chars, stoi, itos, encode, decode

def split_data(data_tensor):
    n = int(0.9 * len(data_tensor))
    return data_tensor[:n], data_tensor[n:]