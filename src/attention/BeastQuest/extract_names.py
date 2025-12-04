import re
from utils import get_text 

def get_names(text):
    names = []
    for line in text.splitlines()[1:]:
        name = re.sub(r"^\d+\.\s*", "", line)
        names.append(name)

    return names


def write_names_to_file(names, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')


if __name__ == "__main__":
    # text = get_text('names.txt')
    # names = get_names(text)
    # write_names_to_file(names, 'clean_names.txt')

    text = get_text('full_names.txt')
    names = [name.split(' ')[0] for name in text.splitlines()]
    write_names_to_file(names, 'short_names.txt')
