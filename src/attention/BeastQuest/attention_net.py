import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from utils import get_text

BATCH_SIZE = 16
EPOCHS = 2000
LEARNING_RATE = 1e-3

# Special tokens
BOS = "<BOS>"
EOS = "<EOS>"
PAD = "<PAD>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_line(line):
    return list(line.upper().strip())


def tokenize_names(names):
    token_counts = Counter()
    tokenized_names = []

    for name in names:
        tokens = tokenize_line(name)
        tokenized_names.append(tokens)
        token_counts.update(tokens)

    return tokenized_names, token_counts


def encode(tokens, token_to_index):
    """ Encode a list of tokens into their corresponding indices, adding BOS and EOS tokens. """
    return [token_to_index[BOS]] + [token_to_index[t] for t in tokens] + [token_to_index[EOS]]


def make_dataset(tokenized_sentences, token_to_index, context_length):
    seqs = []
    for tokens in tokenized_sentences:
        ids = encode(tokens, token_to_index)
        # break long sequences into context_len windows with next-token targets
        # but here sequences are short; we'll pad/truncate to context_len
        if len(ids) > context_length:
            ids = ids[:context_length]
        else:
            # pad
            ids = ids + [token_to_index[PAD]] * (context_length - len(ids))
        seqs.append(ids)
    return seqs


# create simple dataloader sampling random batches (with replacement for simplicity)
def get_batch(data, token_to_index):
    idx = torch.randint(0, data.shape[0], (BATCH_SIZE,))
    x = data[idx]  # (B, T)
    # targets are next-token prediction shifted left: target[t] = x[t+1], last token -> PAD (or EOS)
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = token_to_index[PAD]  # no next token for last position
    return x, y


@torch.no_grad()
def generate(model, prompt, token_to_index, context_length, max_new_tokens=10):
    model.eval()
    index_to_token = {i: tok for tok, i in token_to_index.items()}

    ids = [token_to_index.get(BOS)]

    # encode prompt
    for token in tokenize_line(prompt):
        ids.append(token_to_index.get(token, token_to_index[PAD]))

    # pad or truncate to context_len (keep rightmost tokens)
    if len(ids) < context_length:
        ids = ids + [token_to_index[PAD]] * (context_length - len(ids))
    else:
        ids = ids[-context_length:]

    print("Generating from prompt:", ids)

    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
        logits = model(x)  # (1, T, V)
        # take last token logits
        # if len(ids) < context_len it's okay because we included BOS/pads
        last_logits = logits[0, len(ids)-1]
        probs = F.softmax(last_logits, dim=-1)
        #nxt = torch.argmax(probs).item()
        nxt = torch.multinomial(probs, num_samples=1).item()

        print("Next token:", nxt, "(", index_to_token.get(nxt, "UNK"), ")")

        ids.append(nxt)
        if nxt == token_to_index[EOS]:
            break
        # keep context window size
        ids = ids[-context_length:]

    decoded = [index_to_token[i] for i in ids if i in index_to_token] # and index_to_token[i] not in (BOS, EOS, PAD)]
    return " ".join(decoded)


# ---------- MODEL COMPONENTS ----------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, context_len):
        super().__init__()
        assert head_dim == embed_dim, "for single-head small model we'll set head_dim == embed_dim"
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.out = nn.Linear(head_dim, embed_dim)
        self.scale = 1.0 / math.sqrt(head_dim)
        # causal mask precomputed: (T,T) with -inf on future positions
        mask = torch.tril(torch.ones(context_len, context_len)).unsqueeze(0)  # (1, T, T)
        self.register_buffer("mask", mask)  # 1 means allowed

    def forward(self, x):
        """
        x: (B, T, C)
        Returns: (B, T, C)
        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_dim)
        q = self.query(x)
        v = self.value(x)
        # compute attention scores
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, T, T)
        # apply causal mask: set -inf to future positions
        mask = self.mask[:, :T, :T]  # (1, T, T)
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn_logits, dim=-1)  # (B, T, T)
        out = torch.matmul(attn, v)  # (B, T, head_dim)
        out = self.out(out)  # (B, T, C)
        return out


class SimpleBlock(nn.Module):
    def __init__(self, embed_dim, head_dim, ffn_hidden, context_len):
        super().__init__()
        self.attn = SingleHeadSelfAttention(embed_dim, head_dim, context_len)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_length, ffn_hidden=32):
        super().__init__()
        HEAD_DIM = embed_dim  # single head

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.block = SimpleBlock(embed_dim, HEAD_DIM, ffn_hidden, context_length)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)  # predict logits over vocab

    def forward(self, idx):
        """
        idx: (B, T) token ids
        returns logits (B, T, V)
        """
        B, T = idx.shape
        tok = self.token_emb(idx)             # (B, T, C)
        # Positional embedding
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]  # (1, T, C)
        x = tok + pos
        x = self.block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits


def train(model, data, token_to_index):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_index[PAD])

    print("Starting training on device:", device)
    for step in range(EPOCHS):
        model.train()
        xb, yb = get_batch(data, token_to_index)
        logits = model(xb)  # (B, T, V)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B*T, V), yb.view(B*T))
        optimizer.zero_grad()
        loss.backward()
        # small grad clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f}")

    return model


def main(filename):
    EMBED_DIM = 8

    # Convert list of names to list of tokens
    names = get_text(filename).splitlines()
    max_name_length = max(len(name) for name in names)
    tokenized_names, token_counts = tokenize_names(names)
    # print("Tokenized Names:", tokenized_names)
    # print("Token Counts:", token_counts)

    vocab = [PAD, BOS, EOS] + list(token_counts)
    vocab_size = len(vocab)
    token_to_index = {token: i for i, token in enumerate(vocab)}

    data = make_dataset(tokenized_names, token_to_index, max_name_length + 1)
    data = torch.tensor(data, dtype=torch.long).to(device)

    context_length = max_name_length + 1
    model = TinyLM(vocab_size, EMBED_DIM, context_length).to(device)
    train(model, data, token_to_index)

    gen_name = generate(model, "", token_to_index, context_length, max_new_tokens=10)
    print("Generated name:", gen_name)


if __name__ == "__main__":
    main('short_names.txt')
