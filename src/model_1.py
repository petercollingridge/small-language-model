""" Start testing with a simple example of two, very similar sentences. """


from utils.text import (
    get_hot_one_encoding,
    get_word_pair_vectors,
    get_vocab_list,
    tokenise_sentence,
)

sentences = [
    "sheep are meek",
    "sheep are herbivores",
]

sentence_tokens = [tokenise_sentence(sentence) for sentence in sentences]
# print(sentence_tokens)

vocab = get_vocab_list(sentence_tokens)
# print(vocab)

encode_word = get_hot_one_encoding(vocab)

input_vectors, output_vectors = get_word_pair_vectors(sentence_tokens, encode_word)
print(input_vectors)
print(output_vectors)
