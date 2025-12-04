import torch
import torch.nn as nn
import torch.nn.functional as F

seqs = ['TAT', 'GAG', 'CCT']
seqs = ['ABC', 'ABD']

LEARNING_RATE = 1e-3

# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

chars = sorted(list(set(''.join(seqs))))
vocab = [BOS, PAD, EOS] + list(chars)
vocab_size = len(vocab)

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s): return [stoi[BOS]] + [stoi[c] for c in s] + [stoi[EOS]]
def decode(l): return ''.join([itos[i] for i in l])

# print(encode(seqs[0]))
# print(decode(encode(seqs[0])))

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T) if targets is not None else None
            loss = F.cross_entropy(logits, targets) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


def get_batch():
    x = torch.stack([torch.tensor(encode(seq)[:-1], dtype=torch.long) for seq in seqs])  # (B, T)
    y = torch.stack([torch.tensor(encode(seq)[1:], dtype=torch.long) for seq in seqs])  # (B, T)
    return x, y

# x = torch.tensor(encode('TAT'), dtype=torch.long).unsqueeze(0)  # (1, 3) 
xb, yb = get_batch()  # (B, T)
batch_size, block_size = xb.shape

# print("xb shape:", xb.shape)
# print("xb:", decode(xb[0].tolist()))
# print("yb:", decode(yb[0].tolist()))

# Create model
model = BigramLanguageModel(vocab_size)
# logits, loss = model(xb, yb)
# print(logits.shape)  # (1, 3, vocab_size)
# print(loss)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

for step in range(10000): # increase number of steps for good results...
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step}, loss {loss.item():.4f}")

print(loss.item())

# Generate a sentence starting with <BOS>
start_idx = torch.tensor([[stoi[BOS]]], dtype=torch.long)
print(start_idx)
print(decode(model.generate(idx=start_idx, max_new_tokens=5)[0].tolist()))

# print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)[0].tolist()))

# probs = F.softmax(logits, dim=-1)
# next_token = torch.multinomial(probs[0, -1], num_samples=1)
# print("Next token:", next_token.item(), "(", itos[next_token.item()], ")")
