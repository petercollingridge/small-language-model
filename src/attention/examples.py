from BigramModel import BigramLanguageModel, BigramLanguageModelWithPositionalEncoding
from utils import get_all_seqs, get_text, get_seqs, get_random_seqs, run_model, Tokeniser


def bigram_example(filepath):
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
    # model = BigramLanguageModel(tokeniser.vocab_size)
    model = BigramLanguageModelWithPositionalEncoding(tokeniser.vocab_size, tokeniser.block_size, n_embed=8)

    run_model(model, get_batch)

    # Generate some text
    print(model.generate(tokeniser, max_new_tokens=20))
    print(model.generate(tokeniser, max_new_tokens=20))


if __name__ == "__main__":
    # bigram_example("DNA/data.txt")
    # bigram_example("BeastQuest/short_names.txt")
    bigram_example("BeastQuest/full_names.txt")
