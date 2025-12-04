def get_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text