# ---------- BUILD VOCAB ----------
# naive whitespace tokenizer, build vocabulary

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from torch import device

VOCAB_MIN_FREQ = 1
CONTEXT_LENGTH = 5
EMBED_DIM = 64           # token embedding size
HEAD_DIM = 64            # for single head, head_dim == embed_dim typically
FFN_HIDDEN = 128
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
EPOCHS = 2000

# special tokens
PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentences = [
    "sheep are meek",
    "sheep are herbivores",
    "sheep eat grass",
]


def tokenize_sentence(sentence):
    token_counts = Counter()
    tokenized_sentences = []

    for s in sentences:
        tokens = s.strip().split()
        tokenized_sentences.append(tokens)
        token_counts.update(tokens)

    return tokenized_sentences, token_counts


def encode(tokens, token_to_index):
    """ Encode a list of tokens into their corresponding indices, adding BOS and EOS tokens. """
    return [token_to_index[BOS]] + [token_to_index[t] for t in tokens] + [token_to_index[EOS]]


def make_dataset(tokenized_sentences, token_to_index):
    seqs = []
    for tokens in tokenized_sentences:
        ids = encode(tokens, token_to_index)
        # break long sequences into context_len windows with next-token targets
        # but here sequences are short; we'll pad/truncate to context_len
        if len(ids) > CONTEXT_LENGTH:
            ids = ids[:CONTEXT_LENGTH]
        else:
            # pad
            ids = ids + [token_to_index[PAD]] * (CONTEXT_LENGTH - len(ids))
        seqs.append(ids)
    return torch.tensor(seqs, dtype=torch.long)


# create simple dataloader sampling random batches (with replacement for simplicity)
def get_batch(data, token_to_index):
    idx = torch.randint(0, data.shape[0], (BATCH_SIZE,))
    x = data[idx]  # (B, T)
    # targets are next-token prediction shifted left: target[t] = x[t+1], last token -> PAD (or EOS)
    y = x.clone()
    y[:, :-1] = x[:, 1:]
    y[:, -1] = token_to_index[PAD]  # no next token for last position
    return x, y


# ---------- MODEL COMPONENTS ----------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim, context_len):
        super().__init__()
        assert head_dim == embed_dim, "for single-head small model we'll set head_dim == embed_dim"
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.to_q = nn.Linear(embed_dim, head_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, head_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, head_dim, bias=False)
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
        q = self.to_q(x)  # (B, T, head_dim)
        k = self.to_k(x)
        v = self.to_v(x)
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
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)
        self.block = SimpleBlock(EMBED_DIM, HEAD_DIM, FFN_HIDDEN, CONTEXT_LENGTH)
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size, bias=False)  # predict logits over vocab

    def forward(self, idx):
        """
        idx: (B, T) token ids
        returns logits (B, T, V)
        """
        B, T = idx.shape
        tok = self.token_emb(idx)             # (B, T, C)
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


@torch.no_grad()
def generate(model, prompt, token_to_index, max_new_tokens=10):
    model.eval()
    # simple tokenizer for prompt string
    toks = prompt.strip().split()
    ids = [token_to_index.get(BOS)]
    for t in toks:
        ids.append(token_to_index.get(t, token_to_index[PAD]))
    # pad or truncate to context_len (keep rightmost tokens)
    ids = ids[-CONTEXT_LENGTH:]
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
        logits = model(x)  # (1, T, V)
        # take last token logits
        # if len(ids) < context_len it's okay because we included BOS/pads
        last_logits = logits[0, len(ids)-1]
        probs = F.softmax(last_logits, dim=-1)
        nxt = torch.argmax(probs).item()
        ids.append(nxt)
        if nxt == token_to_index[EOS]:
            break
        # keep context window size
        ids = ids[-CONTEXT_LENGTH:]

    index_to_token = {i: tok for tok, i in token_to_index.items()}
    # decode (skip BOS)
    decoded = [index_to_token[i] for i in ids if i in index_to_token and index_to_token[i] not in (BOS, PAD)]
    return " ".join(decoded)


def main():
    tokenized_sentences, token_counts = tokenize_sentence(sentences)
    vocab = [PAD, BOS, EOS] + [t for t, c in token_counts.items() if c >= VOCAB_MIN_FREQ]
    vocab_size = len(vocab)
    token_to_index = {token: i for i, token in enumerate(vocab)}

    data = make_dataset(tokenized_sentences, token_to_index).to(device)  # shape (N, context_len)
    print(data)

    model = TinyLM(vocab_size).to(device)
    train(model, data, token_to_index)

    # Try generating
    print("Sample generation:", generate(model, "sheep", token_to_index, max_new_tokens=5))
    print("Sample generation:", generate(model, "sheep", token_to_index, max_new_tokens=5))
    print("Sample generation:", generate(model, "sheep", token_to_index, max_new_tokens=5))
    print("Sample generation:", generate(model, "sheep", token_to_index, max_new_tokens=5))

if __name__ == "__main__":
    main()
