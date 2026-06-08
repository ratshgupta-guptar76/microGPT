import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import GPTConfig

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Only supports mutli-head attention
        self.query = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.key = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, config.head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones((config.block_size, config.block_size))))
        self.head_size = config.head_size

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0,  float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
    
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_heads = MultiHeadAttention(config)
        self.ffw = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x

class GPT(nn.Module):
    """GPT module consisting of multiple transformer blocks"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # x is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_nxt = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_nxt), dim=1) # (B, T+1)
        return idx