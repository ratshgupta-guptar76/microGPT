from model import GPT
import torch

ckpt = torch.load('checkpoints/best.pt', weights_only=False)
config = ckpt['config']
m = GPT(config).to(config.device)
m.load_state_dict(ckpt['model'])
m.eval()

# need encode/decode — rebuild vocab from the data file
from train import decode  # works now that training is behind __main__ guard

context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))