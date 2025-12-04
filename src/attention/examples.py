import torch

from BigramModel import BigramLanguageModel
from utils import get_batch, get_text, run_model, Tokeniser


def DNA_example():
    data = get_text("DNA/data.txt")
    seqs = [name.upper().strip() for name in data.splitlines()]
    tokeniser = Tokeniser(''.join(seqs))

    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    xb, yb = get_batch(encoded_seqs)

    # Create model
    model = BigramLanguageModel(tokeniser.vocab_size)

    run_model(model, xb, yb)

    start_idx = torch.zeros((1, 1), dtype=torch.long)
    print(tokeniser.decode(model.generate(idx=start_idx, max_new_tokens=5)[0].tolist()))

if __name__ == "__main__":
    DNA_example()
