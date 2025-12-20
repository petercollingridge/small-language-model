from utils import get_random_seqs


def test_get_random_seqs():
    """
    Given a list of sequences, test the get_random_seqs function.
    It should return random batches of sequences shifted by one for the targets.
    """

    seqs = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20],
    ]
    batch_size = 2
    get_batch = get_random_seqs(seqs, batch_size)

    print(get_batch())


if __name__ == "__main__":
    test_get_random_seqs()
