import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        """Initialize a single attention head.

        Args:
            head_size (int): Dimension of the attention head
        """
        super().__init__()
        self.key = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        """Forward pass for single attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)  

        wei = (
            q @ k.transpose(-2, -1) * C ** -0.5
        )  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) ---> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        """Initialize multi-head attention.

        Args:
            num_heads (int): Number of attention heads
            head_size (int): Dimension of each attention head
        """
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        """Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, N_EMBD):
        """Initialize feed-forward network.

        Args:
            N_EMBD (int): Embedding dimension
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        """Forward pass for feed-forward network."""
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        """Initialize transformer block.

        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
        """

        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Forward pass for transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """Initialize the Bigram Language Model.

        Args:
            vocab_size (int): Size of the vocabulary
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBD)
        self.position_embedding_table = nn.Embedding(
            config.BLOCK_SIZE, config.N_EMBD
        )
        self.blocks = nn.Sequential(
            *[Block(config.N_EMBD, n_head=config.N_HEAD) for _ in range(config.N_LAYER)]
        )
        self.ln_f = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(
            config.N_EMBD, vocab_size
        )

    def forward(self, idx, targets=None):
        """Forward pass for the language model.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T)
            targets (torch.Tensor, optional): Target token indices of shape (B, T)

        Returns:
            tuple: (logits, loss) where logits has shape (B, T, vocab_size)
                   and loss is None if targets not provided
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(
            idx
        )
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=config.get_device())
        )  # (T, C)
        x = (
            tok_emb + pos_emb
        )
        x = self.blocks(
            x
        )
        x = self.ln_f(
            x
        )
        logits = self.lm_head(
            x
        )

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new tokens autoregressively.

        Args:
            idx (torch.Tensor): Starting context of shape (B, T)
            max_new_tokens (int): Number of new tokens to generate

        Returns:
            torch.Tensor: Extended sequence of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
