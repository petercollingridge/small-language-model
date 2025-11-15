sentences = [
    "sheep are meek",
    "sheep are herbivores",
    "sheep eat grass",
    "sheep like jumping",
]

tokens = ['<>', 'are', 'eat', 'grass', 'herbivores', 'jumping', 'like', 'meek', 'sheep']

vector = [
    [-2.7752,  5.7288,  6.3385,  1.9580,  1.9580,  1.9580, -3.8554,  1.9580, -5.3846],
    [-4.3258, -0.4468,  5.7416, -5.0142, -5.0142, -5.0142,  6.5577, -5.0142, 1.8858]]

transitions = {
    '<>': {'sheep': 1.0},
    'are': {'herbivores': 0.49999142, 'meek': 0.5000086},
    'eat': {'grass': 1.0},
    'grass': {'<>': 1.0},
    'herbivores': {'<>': 1.0},
    'jumping': {'<>': 1.0},
    'like': {'jumping': 1.0},
    'meek': {'<>': 1.0},
    'sheep': {'are': 0.50001776, 'eat': 0.24999303, 'like': 0.24998921},
}
