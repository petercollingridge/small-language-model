sentences = [
    "sheep are meek",
    "sheep are herbivores",
    "sheep eat grass",
]

tokens = ['<>', 'are', 'eat', 'grass', 'herbivores', 'meek', 'sheep']

embedding = [
    [ 1.3612,  0.8550,  5.6152, -1.8107, -1.8107, -1.8107, -5.4516],
    [-5.2350,  3.6278,  0.1614, -4.5363, -4.5363, -4.5363,  2.5408]
]

transitions = {
    '<>': {'sheep': 1.0},
    'are': {'herbivores': 0.5, 'meek': 0.5},
    'eat': {'grass': 1.0},
    'grass': {'<>': 1.0},
    'herbivores': {'<>': 1.0},
    'meek': {'<>': 1.0},
    'sheep': {'are': 0.6679165, 'eat': 0.3320835},
}
