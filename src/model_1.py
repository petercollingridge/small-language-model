""" Start testing with a simple example of two, very similar sentences. """

import os
from utils.neural_net import train_simple_model
from utils.text import (
    get_hot_one_encoding,
    get_word_pair_vectors,
    get_vocab_list,
    tokenise_sentence,
)
from vis.svg_model import generate_svg, write_svg_to_file

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

model = train_simple_model(input_vectors, output_vectors)

model_svg = generate_svg(model, labels=vocab)
write_svg_to_file(model_svg, os.path.join('src', 'vis','test.svg'))
