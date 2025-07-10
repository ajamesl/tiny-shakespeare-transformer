import torch
from tqdm import tqdm
from src import config, model as M, utils

text, chars, stoi, itos, encode, decode = utils.load_data()
vocab_size = len(chars)
data = torch.tensor(encode(text), dtype=torch.long)
train_data, val_data = utils.split_data(data)

device = config.get_device()
model = M.BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data_source[i:i + config.BLOCK_SIZE] for i in ix]).to(device)
    y = torch.stack([data_source[i + 1:i + config.BLOCK_SIZE + 1] for i in ix]).to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.EVAL_ITERS)
        for k in range(config.EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in tqdm(range(config.MAX_ITERS)):
    if iter % config.EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "checkpoints/tiny_shakespeare_final.pt")