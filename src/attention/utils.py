import torch

# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

LEARNING_RATE = 1e-3

class Tokeniser:
    """ Class that converts characters to integers and back """

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab = [BOS, PAD, EOS] + list(self.chars)
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[BOS]] + [self.stoi[char] for char in text] + [self.stoi[EOS]]

    def decode(self, lst):
        end = lst.index(self.stoi[EOS]) if self.stoi[EOS] in lst else len(lst)
        lst = lst[1:end]  # remove BOS and everything after EOS
        chars = [self.itos[i] for i in lst if i != self.stoi[PAD]]
        return ''.join(chars)


def get_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_batch(seqs):
    x = torch.stack([torch.tensor(seq[:-1], dtype=torch.long) for seq in seqs])  # (B, T)
    y = torch.stack([torch.tensor(seq[1:], dtype=torch.long) for seq in seqs])  # (B, T)
    return x, y


def run_model(model, xb, yb, steps=10000):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(steps):
        # Evaluate the loss
        logits, loss = model(xb, yb)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}, loss {loss.item():.4f}")
