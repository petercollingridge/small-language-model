import html

def get_attribute_string(attrs):
    """
    Convert a dictionary of attributes to a string.
    """
    return " ".join(f'{key}="{value}"' for key, value in attrs.items())


def wrap_in_group(string, attrs=None):
    """
    Wrap a string in a <g> tag.
    """

    attr_string = "" if attrs is None else  get_attribute_string(attrs)
    return f'<g {attr_string}>{string}\n</g>'


def text_element(text, attrs=None):
    """
    Wrap a string in a <text> tag.
    """

    attr_string = "" if attrs is None else get_attribute_string(attrs)
    return f'<text {attr_string}>{html.escape(text)}</text>'
