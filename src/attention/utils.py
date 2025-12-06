import torch

# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

LEARNING_RATE = 1e-3

class Tokeniser:
    """ Class that converts characters to integers and back """

    def __init__(self, seqs):
        self.block_size = max(len(seq) for seq in seqs)  # context size
        self.chars = sorted(list(set(''.join(seqs))))
        self.vocab = [BOS, PAD, EOS] + list(self.chars)
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text):
        encoded_text = [self.stoi[char] for char in text]
        if len(encoded_text) < self.block_size:
            encoded_text += [self.stoi[PAD]] * (self.block_size - len(encoded_text))
        return [self.stoi[BOS]] + encoded_text + [self.stoi[EOS]]

    def decode(self, lst):
        end = lst.index(self.stoi[EOS]) if self.stoi[EOS] in lst else len(lst)
        lst = lst[1:end]  # remove BOS and everything after EOS
        chars = [self.itos[i] for i in lst if i != self.stoi[PAD]]
        return ''.join(chars)


def get_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_seqs(text):
    """ Given a block of text, return a list of sequences (lines) """

    seqs = [line.strip() for line in text.splitlines() if line.strip()]
    return seqs


def get_all_seqs(seqs):
    """
    Given a list of vectors, return a function that returns all the vectors as tensors,
    shifted by one for the targets
    B: batch size, i.e. number of sequences
    T: time steps, i.e. length of each sequence - 1
    """

    def get_batch():
        x = torch.stack([torch.tensor(seq[:-1], dtype=torch.long) for seq in seqs])  # (B, T)
        y = torch.stack([torch.tensor(seq[1:], dtype=torch.long) for seq in seqs])  # (B, T)
        return x, y
    return get_batch


def run_model(model, get_batch, steps=10000):
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(steps):
        # Get a batch of data
        inputs, targets = get_batch()

        # Evaluate the loss
        logits, loss = model(inputs, targets)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"Step {step}, loss {loss.item():.4f}")
