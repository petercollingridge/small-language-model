import torch
import torch.nn as nn
import torch.nn.functional as F
import math

text = "hello world"

# Build a character-level vocabulary
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

# print(chars)
# print(stoi)
# print(itos)

def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# print(data)  # tensor of character indices

vocab_size = len(chars)
embed_dim = 16
block_size = 8  # how many characters context we look at


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SimpleAttentionBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        mask = torch.tril(torch.ones_like(attn_scores)).bool()
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V
        x = self.ln1(x + attn_output)
        x = self.ln2(x + self.ff(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = PositionalEncoding(embed_dim, block_size)
        self.block = SimpleAttentionBlock(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx)
        x = self.pos_emb(x)
        x = self.block(x)
        logits = self.lm_head(x)
        return logits


model = TinyTransformer(vocab_size, embed_dim, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# print(model)

def get_batch():
    ix = torch.randint(len(data) - block_size, (1,))
    x = data[ix:ix + block_size]
    y = data[ix + 1:ix + block_size + 1]
    return x.unsqueeze(0), y.unsqueeze(0)

for step in range(3000):
    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, loss {loss.item():.4f}")


model.eval()
context = torch.tensor([[stoi['h']]], dtype=torch.long)

for _ in range(20):
    logits = model(context[:, -block_size:])
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    context = torch.cat([context, next_id], dim=1)

print(decode(context[0].tolist()))
