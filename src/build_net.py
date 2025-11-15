""" Start testing with a simple example of two, very similar sentences. """

import os
import torch

from utils.neural_net import train_simple_model, run_model
from utils.text import (
    get_hot_one_encoding,
    get_word_pair_vectors,
    get_vocab_list,
    tokenise_sentence,
)
from vis.svg_model import NeuralNetSVG


def get_sentences(folder):
    """ Load sentences from a text file in the given folder. """

    sentences_file = os.path.join(folder, 'sentences.txt')
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def build_model(sentences, token_list, loop_n=100000):
    """
    Build a simple neural network model for the given sentences.
    """

    print("Building model...")

    # Get the one-hot encoding for each word in the vocabulary
    encode_word = get_hot_one_encoding(token_list)

    # Get input and output vectors for the word pairs
    input_vectors, output_vectors = get_word_pair_vectors(sentences, encode_word)

    # Train the model
    model = train_simple_model(input_vectors, output_vectors, hidden_size=2, loop_n=loop_n)

    return model


def save_model(model, folder, filename = "model.pt"):
    """
    Save the trained model to a file.
    """

    filepath = os.path.join(folder, filename)
    print(f"Saving model to {filepath}...")
    torch.save(model.state_dict(), filepath)


def save_token_list(filepath, token_list):
    """
    Save the token list to a text file.
    """

    with open(filepath, 'a', encoding='utf-8') as f:
        # Token list
        f.write("\ntokens = [")
        f.write(", ".join(f"'{token}'" for token in token_list))
        f.write("]\n")


def save_model_parameters(filepath, model):
    """
    Save the model data in a human-readable format in a Python file,
    so data can be analysed.
    """

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write("\nmodel_parameters = {\n")
        for name, param in model.named_parameters():
            name_parts = name.split('.')
            name = f"{name_parts[1]}{name_parts[0]}"
            f.write(f"    '{name}': {param.data.tolist()},\n")
        f.write("}\n")


def save_token_embeddings(filepath, token_list, model):
    """
    Save the token embeddings to a file.
    """

    # Get the embedding layer from the model
    embedding_layer = model[0]
    column1 = embedding_layer.weight.data[0].tolist()
    column2 = embedding_layer.weight.data[1].tolist()

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write("\ntoken_embeddings = {\n")
        for i, token in enumerate(token_list):
            f.write(f"    '{token}': [{column1[i]}, {column2[i]}],\n")
        f.write("}\n")


def save_transition_probabilities(filepath, token_list, model):
    """
    Save a dictionary mapping each token to a dictionary 
    of possible next tokens and their probabilities.
    """

    # Get square identity matrix as test inputs, representing each token in the vocabulary
    test_inputs = torch.eye(len(token_list))
    test_outputs = run_model(model, test_inputs)

    with open(filepath, 'a', encoding='utf-8') as f:
        f.write("\ntransitions = {")
        for i, input_word in enumerate(token_list):
            output_distribution = {
                token_list[j]: probability for j, probability in enumerate(test_outputs[i])
                if probability > 0
            }
            f.write(f"\n    '{input_word}': {output_distribution},")
        f.write("\n}\n")


def draw_neural_net_svg(model, token_list, folder, filename='neural_net.svg'):
    """
    Draw an SVG representation of the neural network model.
    """

    filepath = os.path.join(folder, filename)
    test_inputs = torch.eye(len(token_list))
    model_svg = NeuralNetSVG(model, labels=token_list)
    model_svg.write_to_file(filepath, test_inputs)


def main(folder):
    """
    Get sentences from the given folder, build and train a model,
    then save the model data in the same folder.
    """

    # Load sentences from file
    sentences = get_sentences(folder)
    sentences_as_tokens = [tokenise_sentence(sentence) for sentence in sentences]
    token_list = get_vocab_list(sentences_as_tokens)

    model = build_model(sentences_as_tokens, token_list, loop_n=10000)
    # save_model(model, folder)

    save_file = os.path.join(folder, 'model_data.py')
    # Clear the file if it already exists
    open(save_file, 'w', encoding='utf-8').close()
    save_token_list(save_file, token_list)
    save_model_parameters(save_file, model)
    save_token_embeddings(save_file, token_list, model)
    save_transition_probabilities(save_file, token_list, model)

    # Generate SVGs
    draw_neural_net_svg(model, token_list, folder)


if __name__ == "__main__":
    FOLDER = 'example-1'
    main(FOLDER)
