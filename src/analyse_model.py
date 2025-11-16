import os
import sys
import torch

from utils.io import get_sentences, load_model
from utils.text import get_vocab_list, tokenise_sentence
from vis.draw_network import NeuralNetSVG
from vis.plot_embeddings import plot_embeddings


def draw_neural_net_svg(model, token_list, folder, filename='neural_net.svg'):
    """
    Draw an SVG representation of the neural network model.
    """

    filepath = os.path.join(folder, filename)
    test_inputs = torch.eye(len(token_list))
    model_svg = NeuralNetSVG(model, labels=token_list)
    model_svg.write_to_file(filepath, test_inputs)


def draw_embeddings_chart(model, tokens, folder, filename='embeddings.svg'):
    """
    Draw an SVG chart of the word embeddings.
    """

    filepath = os.path.join(folder, filename)
    embeddings = model[0].weight

    svg_content = plot_embeddings(tokens, embeddings)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_content)


def main(folder):
    """
    Open a model and generate SVG visualisations of it.
    """

    # Load sentences from file
    sentences = get_sentences(folder)
    sentences_as_tokens = [tokenise_sentence(sentence) for sentence in sentences]
    token_list = get_vocab_list(sentences_as_tokens)

    # Load model parameters
    model = load_model(folder)

    # Generate SVGs
    draw_neural_net_svg(model, token_list, folder)

    draw_embeddings_chart(model, token_list, folder)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main('example1')
