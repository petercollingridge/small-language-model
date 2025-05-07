sentences = [
    "sheep are meek",
    "sheep are herbivores",
]

tokens = ['<>', 'are', 'herbivores', 'meek', 'sheep']

embedding = [
    [-2.1476, -2.4123, 2.9056, 2.9068, 4.0996],
    [3.4866, -2.8813, 2.8036, 2.8050, -3.5083]
]

transitions = {
    '<>': {'sheep': 1.0},
    'are': {'herbivores': 0.49998927, 'meek': 0.5000107},
    'herbivores': {'<>': 1.0},
    'meek': {'<>': 1.0},
    'sheep': {'are': 1.0},
}
