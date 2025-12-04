from collections import Counter, defaultdict
from itertools import count
from utils import get_text


def count_letters(names):
    letter_counts = Counter()
    for name in names:
        for letter in name:
            letter_counts[letter] += 1
    return letter_counts


def count_letter_pairs(names):
    pair_counts = Counter()
    for name in names:
        for i in range(len(name) - 1):
            pair = name[i: i + 2]
            pair_counts[pair] += 1
    return pair_counts


def get_next_letter_counts(names):
    next_letter_counts = defaultdict(Counter)
    for name in names:
        for i in range(len(name) - 1):
            letter = name[i]
            next_letter = name[i + 1]
            next_letter_counts[letter][next_letter] += 1
    return next_letter_counts


def get_next_letter_frequencies(next_letter_counts):
    next_letter_frequencies = {}
    for letter, counter in next_letter_counts.items():
        total = sum(counter.values())
        # frequencies = {next_letter: count / total for next_letter, count in counter.items()}
        # next_letter_frequencies[letter] = frequencies

        for next_letter, count in counter.items():
            next_letter_frequencies[letter + next_letter] = count / total

    return next_letter_frequencies


def get_tokens(names, max_tokens=26):
    single_letters = count_letters(names)
    letter_pairs = count_letter_pairs(names)

    tokens = set()
    for name in names:
        for letter in name:
            tokens.add(letter)
    return sorted(tokens)


def tokenise_names(names):
    tokenised_names = []
    for name in names:
        tokenised_name = name.split()
        tokenised_names.append(tokenised_name)
    return tokenised_names


if __name__ == "__main__":
    text = get_text('short_names.txt')
    names = [name.upper().strip() for name in text.splitlines()]
    # counts = count_letters(names)
    # counts = count_letter_pairs(names)
    counts = get_next_letter_counts(names)
    counts = get_next_letter_frequencies(counts)

    for item, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"'{item}': {count:.4f}")

