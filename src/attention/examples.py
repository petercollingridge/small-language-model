import torch

from AttentionModel import BigramWithAttentionModel
from BigramModel import BigramLanguageModel, BigramLanguageModelWithPositionalEncoding
from utils import generate_text, get_all_seqs, get_text, get_seqs, get_random_seqs, run_model, Tokeniser


def bigram_example(filepath):
    text = get_text(filepath)
    seqs = get_seqs(text)
    tokeniser = Tokeniser(seqs)

    print(tokeniser.vocab_size, tokeniser.block_size)

    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    # get_batch = get_all_seqs(encoded_seqs)
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)

    x, y = get_batch()
    print(x.shape, y.shape)

    # Create model
    # model = BigramLanguageModel(tokeniser.vocab_size)
    model = BigramLanguageModelWithPositionalEncoding(tokeniser.vocab_size, tokeniser.block_size, n_embed=8)

    run_model(model, get_batch)

    # Generate some text
    max_new_tokens = min(20, tokeniser.block_size)
    generate_text(model, tokeniser, n = 10, max_new_tokens = max_new_tokens)


def attention_example(filepath, embed_dim=8):
    text = get_text(filepath)
    seqs = get_seqs(text)
    tokeniser = Tokeniser(seqs)

    print(tokeniser.vocab_size, tokeniser.block_size)

    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_batch = get_random_seqs(encoded_seqs, batch_size=8)

    x, y = get_batch()
    print(x.shape, y.shape)

    # Create model
    model = BigramWithAttentionModel(tokeniser.vocab_size, tokeniser.block_size, embed_dim)

    run_model(model, get_batch)

    # Generate some text
    generate_text(model, tokeniser, n = 10, max_new_tokens = 20)


if __name__ == "__main__":
    # bigram_example("DNA/data.txt")
    # attention_example("BeastQuest/short_names.txt")
    # bigram_example("BeastQuest/short_names.txt")
    # bigram_example("BeastQuest/full_names.txt")
    attention_example("BeastQuest/full_names.txt", 16)
