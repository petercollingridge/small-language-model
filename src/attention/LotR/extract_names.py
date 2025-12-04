import re


def get_lotr_text(filename='lotr.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def get_names(text):
    names = []
    for line in text.splitlines():
        if line.startswith('#'):
            parts = re.split(r'[,@>–\-]', line, maxsplit=2)
            name = re.sub(r"^#\d+\s*", "", parts[0])
            names.append(name)

    return sorted(set(names))


def write_names_to_file(names, filename='lotr_names.txt'):
    with open(filename, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')


text = get_lotr_text()
names = get_names(text)
write_names_to_file(names)