""" Start testing with a simple example of two, very similar sentences. """

import os
import torch

from utils.neural_net import train_simple_model
from utils.text import (
    get_hot_one_encoding,
    get_word_pair_vectors,
    get_vocab_list,
    tokenise_sentence,
)
from vis.svg_model import NeuralNetSVG

sentences = [
    "sheep are meek",
    "sheep are herbivores",
]

sentence_tokens = [tokenise_sentence(sentence) for sentence in sentences]
# print(sentence_tokens)

vocab = get_vocab_list(sentence_tokens)

encode_word = get_hot_one_encoding(vocab)

input_vectors, output_vectors = get_word_pair_vectors(sentence_tokens, encode_word)
# print(input_vectors)
# print(output_vectors)

model = train_simple_model(input_vectors, output_vectors, hidden_size = 2, loop_n=10000)

test_input = torch.tensor([0, 1.0, 0, 0, 0])

filename = os.path.join('src', 'vis','basic model.svg')
model_svg = NeuralNetSVG(model, labels=vocab)
model_svg.activate(test_input)
model_svg.write_to_file(filename)

# filename = os.path.join('src', 'vis','basic model activation.svg')
# model_svg = NeuralNetActivationSVG(model, activation, labels=vocab)
# model_svg.write_to_file(filename)
