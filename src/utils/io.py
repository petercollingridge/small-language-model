import os
import torch

from utils.neural_net import get_model


def get_model_parameters(folder):
    """ Load a model from a file in the given folder. """

    parameters_file = os.path.join(folder, 'model_data.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_parameters", parameters_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model_parameters


def get_sentences(folder):
    """ Load sentences from a text file in the given folder. """

    sentences_file = os.path.join(folder, 'sentences.txt')
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences


def load_model(folder):
    """ Load a trained model from a file. """

    filepath = os.path.join(folder, 'model.pt')
    print(f"Loading model from {filepath}...")

    # Load trained model
    state_dict = torch.load(filepath, map_location="cpu")
    model_state = torch.load(filepath)

    input_size = model_state['0.weight'].shape[1]
    hidden_size = model_state['0.weight'].shape[0]
    output_size = model_state['1.weight'].shape[0]

    model = get_model(input_size, hidden_size, output_size)
    model.load_state_dict(state_dict)
    model.eval()

    return model
