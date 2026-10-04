import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import sys


class NGram(nn.Module):
    def __init__(self, vocab_size: int, hdim: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(vocab_size, hdim) * 0.1)
        self.W2 = nn.Parameter(torch.randn(hdim, vocab_size) * 0.1)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.W1[x]  # (B, T, hdim)
        h = F.gelu(emb)  # (B, T, hdim)
        logits = h @ self.W2  # (B, T, vocab_size)

        loss = None
        if y is not None:
            # logits: (B, T, vocab_size) -> (B*T, vocab_size); y: (B, T) -> (B*T,).
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        return logits, loss

@torch.no_grad()
def generate(ngram: NGram, ids: torch.Tensor, max_tokens: int = 24) -> torch.Tensor:
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)   # (1, T)

    for _ in range(max_tokens):
        logits, _ = ngram.forward(ids)  # (B, T, vocab_size)
        last_logits = logits[:, -1, :]  # (B, vocab_size)
        probs = F.softmax(last_logits, dim=-1)  # (B, vocab_size)
        next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
        ids = torch.cat([ids, next_id], dim=1)   # (B, T+1)

    return ids

if __name__ == "__main__":
    print(f"python: {sys.version}")
    print(f"torch: {torch.version.__version__}")

    with open('datasets/willy.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size: int = len(chars)  # or "classification bins"
    print(''.join(chars))
    print(vocab_size)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s]
    def decode(l: list[int]) -> str:
        return ''.join([itos[i] for i in l])

    print(encode("Hello world"))
    print(decode(encode("Hello world")))

    data: torch.Tensor = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    print(f"data length: {data.shape[0]}")

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    has_cuda: bool = torch.cuda.is_available()
    device_str: str = "cuda" if has_cuda else "cpu"
    device: torch.device = torch.device(device_str)
    print(f"has cuda {has_cuda} torch device: {device}")

    seed = 1337
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    sequence_len: int = 8
    hdim: int = 16  # hidden dimension
    lr: float = 0.2
    reg: float = 0.01

    ngram: NGram = NGram(vocab_size, hdim)

    ngram = ngram.to(device)
    train_data = train_data.to(device)
    ngram.train()

    max_steps = (data.shape[0] - sequence_len) // sequence_len
    print(f"max steps: {max_steps}")

    for i in range(max_steps):
        x = train_data[i:i + sequence_len]  # (T, )
        y = train_data[i + 1:i + sequence_len + 1]  # (T, )

        x = x.unsqueeze(0)  # (B, T)
        y = y.unsqueeze(0)  # (B, T)

        logits, loss = ngram.forward(x, y)

        loss = loss + reg * (ngram.W1 ** 2).mean() + reg * (ngram.W2 ** 2).mean()

        if (i < 5 or i % 1000 == 0):
            print(f"loss: %.4f step %d " % (loss.data.item(), i)) #  causes sync from gpu to cpu
        loss.backward()

        for p in ngram.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad.data
                p.grad.data.zero_()


    print("running val...")
    ngram.eval()
    val_loss = 0.0
    count = 0
    val_data = val_data.to(device)
    with torch.no_grad():
        for i in range(0, len(val_data) - sequence_len - 1, sequence_len):
            x = val_data[i:i + sequence_len]
            y = val_data[i + 1:i + sequence_len + 1]

            x = x.unsqueeze(0)  # (B, T)
            y = y.unsqueeze(0)  # (B, T)

            _, loss = ngram.forward(x, y)

            val_loss += loss.item()
            count += 1

    val_loss /= count
    print(f"val loss: {val_loss:.4f}")

    print("generating sample: ")
    inp_x = torch.tensor(encode("Oh Lord, the toilet paper is out"), dtype=torch.long).to(device)
    print(decode(generate(ngram, inp_x, 256)[0].tolist()))
    print(decode(generate(ngram, inp_x, 512)[0].tolist()))
