""" Start testing with a simple example of two, very similar sentences. """

import os
import torch

from utils.neural_net import train_simple_model, run_model
from utils.text import (
    get_hot_one_encoding,
    get_word_pair_vectors,
    get_vocab_list,
    output_transition_probabilities,
    tokenise_sentence,
)
from vis.svg_model import NeuralNetSVG
from data.example1a import sentences

def build_model(sentences, loop_n=100000):
    """
    Build a simple neural network model for the given sentences.
    """
    # Tokenize the sentences
    sentence_tokens = [tokenise_sentence(sentence) for sentence in sentences]

    # Get the vocabulary list
    vocab = get_vocab_list(sentence_tokens)

    # Get the one-hot encoding for each word in the vocabulary
    encode_word = get_hot_one_encoding(vocab)

    # Get input and output vectors for the word pairs
    input_vectors, output_vectors = get_word_pair_vectors(sentence_tokens, encode_word)

    # Train the model
    model = train_simple_model(input_vectors, output_vectors, hidden_size=2, loop_n=loop_n)

    return model, vocab

model, vocab = build_model(sentences, loop_n=30000)

test_inputs = torch.eye(len(vocab))
test_outputs = run_model(model, test_inputs)
output_transition_probabilities(vocab, test_outputs)

# filename = os.path.join('src', 'vis','example_1b.svg')
# model_svg = NeuralNetSVG(model, labels=vocab)
# model_svg.write_to_file(filename, test_inputs)
