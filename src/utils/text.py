""" Utility functions for preparing text for a processing. """

import torch


def get_hot_one_encoding(vocab):
    """
    Given a list of words, return a matrix with ones along the diagonal
    and a function that extracts the hot one encoding for a given word.
    """

    # Get matrix with ones along the diagonal
    matrix = torch.eye(len(vocab))
    indicies = {word: idx for idx, word in enumerate(vocab)}

    def get_encoding(word):
        return matrix[indicies[word]]

    return get_encoding


def get_next_words(sentences):
    """
    Given a list of list of tokens,
    return a list of 2-tuples, with one token and the token that follows it.
    e.g. "The next word" => [["The", "next"], ["next", "word"]]
    """

    pairs = []
    for sentence in sentences:
        for ix in range(0, len(sentence) - 1):
            pairs.append((sentence[ix], sentence[ix + 1]))

    return pairs


def get_vocab_list(sentence_tokens):
    """ Given a list of list of tokens, return a sorted list of unique tokens. """

    unique_tokens = set()
    for sentence in sentence_tokens:
        for token in sentence:
            unique_tokens.add(token)

    return sorted(list(unique_tokens))


def get_word_pair_vectors(sentences, encode_word):
    """
    Given a list of sentences, return a list of input words, and a list of words that follow them.
    e.g. "The next word" => input = ["The", "next"] output = ["next", "word"]
    Words in both lists are encoded with the encode_word function.
    """

    input_vectors = []
    output_vectors = []

    for sentence in sentences:
        for ix in range(0, len(sentence) - 1):
            word1 = encode_word(sentence[ix])
            word2 = encode_word(sentence[ix + 1])
            input_vectors.append(word1)
            output_vectors.append(word2)

    return torch.stack(input_vectors), torch.stack(output_vectors)


def tokenise_sentence(sentence):
    """ Split a string into a list of tokens. """
    return f"<> {sentence} <>".split(" ")
