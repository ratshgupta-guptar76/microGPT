import torch
import torch.optim

from model import *
from config import GPTConfig

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 1
batch_size = 64
block_size = 256
train_ratio = 0.9
learning_rate = 3e-4
dropout = 0.2

max_iter = 600000
num_iter = 5000
eval_interval = 1000
n_embed = 384
n_head = 6
n_layer = 6

with open('data/shakespeare.txt', 'r', encoding='utf-8') as FILE:
    text = FILE.read()

tokens = torch.empty(0, dtype=torch.long)
chars = []
stoi = {}
itos = {}
vocab_size = 0
data = torch.empty(0, dtype=torch.long)
train_data = torch.empty(0, dtype=torch.long)
val_data = torch.empty(0, dtype=torch.long)


def build_vocab(text):
    global chars, stoi, itos, vocab_size
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    return chars, stoi, itos


build_vocab(text)

# encode/decode as simple lambdas using global `stoi`/`itos`
# assume inputs are correct (e.g., lists of ints for decode)
encode = lambda e: [stoi[c] for c in e]
decode = lambda d: ''.join([itos[i] for i in d])


data = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(len(data) * train_ratio)
train_data = data[:split_idx]
val_data = data[split_idx:]
tokens = data


def load_data(filepath='../data/shakespeare.txt'):
    global text, tokens, data, train_data, val_data
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    build_vocab(text)
    data = torch.tensor(encode(text), dtype=torch.long)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    tokens = data

    return data


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

config = GPTConfig(
    block_size=block_size,
    batch_size=batch_size,
    train_ratio=train_ratio,
    learning_rate=learning_rate,
    eval_interval=eval_interval,
    eval_iters=eval_iters,
    num_iter=num_iter,
    max_iter=max_iter,
    n_embd=n_embed,
    vocab_size=vocab_size,
    attention_type='multi',
    dropout=dropout,
    n_head=n_head,
    n_layer=n_layer,
)
device = config.device

# TODO: Add the training loop 
model = GPT(config)
m = model.to(device)

# TODO: Add optimizer setup here
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(num_iter):
    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {iter}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))