from vis.utils import text_element


def plot_embeddings(labels, embeddings):
    svg_width = 200
    svg_height = 200
    margin = 10

    # Find min and max values for scaling
    x_values = embeddings[0]
    y_values = embeddings[1]

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    max_value = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    max_range = 2 * max_value

    def scale_x(x):
        return round(float(margin + (x + max_value) / max_range * (svg_width - 2 * margin)), 1)

    def scale_y(y):
        return round(float(svg_height - margin - (y + max_value) / max_range * (svg_height - 2 * margin)), 1)

    # Create SVG elements
    svg_elements = [
      f'<svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">'
    ]

    # Draw axes
    x0 = scale_x(0)
    y0 = scale_y(0)
    svg_elements.append(f'<line x1="{margin}" y1="{y0}" x2="{svg_width - margin}" y2="{y0}" stroke="black"/>')
    svg_elements.append(f'<line x1="{x0}" y1="{margin}" x2="{x0}" y2="{svg_height - margin}" stroke="black"/>')

    # Plot points and labels
    for i, label in enumerate(labels):
        cx = scale_x(x_values[i])
        cy = scale_y(y_values[i])
        svg_elements.append(f'<path d="M{cx - 3} {cy - 3}l7 7m0 -7l-7 7" fill="none" stroke="red"/>')
        svg_elements.append(text_element(label, {'x': cx + 5, 'y': cy - 5, 'font-size': 12, 'fill': 'black'}))

    svg_elements.append('</svg>')

    # Combine and return SVG content
    return "\n".join(svg_elements)


if __name__ == "__main__":
    # points = [
    #   ['<>', -2.1476, 3.4866],
    #   ['are', -2.4123, -2.8813],
    #   ['herbivores', 2.9056, 2.8036],
    #   ['meek', 2.9068, 2.8050],
    #   ['sheep', 4.0996, -3.5083],
    # ]

    # vector = [
    #     [ 1.3612,  0.8550,  5.6152, -1.8107, -1.8107, -1.8107, -5.4516],
    #     [-5.2350,  3.6278,  0.1614, -4.5363, -4.5363, -4.5363,  2.5408]]
    # tokens = ['<>', 'are', 'eat', 'grass', 'herbivores', 'meek', 'sheep']

    vector = [
        [-2.7752,  5.7288,  6.3385,  1.9580,  1.9580,  1.9580, -3.8554,  1.9580, -5.3846],
        [-4.3258, -0.4468,  5.7416, -5.0142, -5.0142, -5.0142,  6.5577, -5.0142, 1.8858]]
    tokens = ['<>', 'are', 'eat', 'grass', 'herbivores', 'jumping', 'like', 'meek', 'sheep']

    svg_content = plot_embeddings(tokens, vector)

    with open("chart-1b.svg", "w") as f:
        f.write(svg_content)
