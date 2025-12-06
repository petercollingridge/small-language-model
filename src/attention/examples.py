from BigramModel import BigramLanguageModel
from utils import get_all_seqs, get_text, get_seqs, run_model, Tokeniser


def DNA_example():
    text = get_text("DNA/data.txt")
    seqs = get_seqs(text)
    tokeniser = Tokeniser(seqs)

    # Create model
    model = BigramLanguageModel(tokeniser.vocab_size)

    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]
    get_batch = get_all_seqs(encoded_seqs)

    run_model(model, get_batch)

    # Generate some text
    print(model.generate(tokeniser))
    print(model.generate(tokeniser))


def beast_quest_example():
    text = get_text("BeastQuest/short_names.txt")
    seqs = get_seqs(text)
    tokeniser = Tokeniser(seqs)

    # Create model
    model = BigramLanguageModel(tokeniser.vocab_size)

    # Function to get batches of training data
    encoded_seqs = [tokeniser.encode(seq) for seq in seqs]

    get_batch = get_all_seqs(encoded_seqs)

    run_model(model, get_batch)

    # Generate some text
    print(model.generate(tokeniser))
    print(model.generate(tokeniser))

if __name__ == "__main__":
    # DNA_example()
    beast_quest_example()
