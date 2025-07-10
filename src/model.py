import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  # Shape: (B, T, C)
        q = self.query(x)  # Shape: (B, T, C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)  # Apply dropout to attention weights
        v = self.value(x)  # Shape: (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multi-heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concatenate the outputs of all heads
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, N_EMBD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(config.DROPOUT), # Optional: dropout for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # N_EMBD: embeddings dimension, N_HEAD: number of heads we want
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # Layer normalization for self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # Layer normalization for feed-forward network

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBD)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)  # Optional: position embeddings
        self.blocks = nn.Sequential(*[Block(config.N_EMBD, n_head=config.N_HEAD) for _ in range(config.N_LAYER)]) # Stack of transformer blocks
        self.ln_f = nn.LayerNorm(config.N_EMBD)  # Final layer normalization
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size)  # Linear layer to project embeddings to logits
         

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B: batch size, T: block size

        # idx and targets are both (batch_size, BLOCK_SIZE) / (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # Shape: (batch_size, BLOCK_SIZE, vocab_size) / (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.get_device()))  # Shape: (BLOCK_SIZE, N_EMBD) / (T, C)
        x = tok_emb + pos_emb # Combine token and position embeddings - Shape: (batch_size, BLOCK_SIZE, N_EMBD) / (B, T, C)
        x = self.blocks(x)  # Pass through transformer blocks - Shape: (batch_size, BLOCK_SIZE, N_EMBD) / (B, T, C)
        x = self.ln_f(x)  # Final layer normalization - Shape: (batch_size, BLOCK_SIZE, N_EMBD) / (B, T, C)
        logits = self.lm_head(x)  # Shape: (batch_size, BLOCK_SIZE, vocab_size) / (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, BLOCK_SIZE) / (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx