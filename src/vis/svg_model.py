import html
import torch.nn.functional as F

STYLES = """
<style>
    circle { fill: lightblue; stroke: white; stroke-width: 1; }
    line { stroke-width: 1; }
    text { font-family: sans-serif; font-size: 12px; fill: black; alignment-baseline: middle; }
</style>\n"""

def generate_svg(
    neural_net,
    labels=[],
    margin_x=100,
    margin_y=10,
    node_radius=20,
    layer_spacing=150,
    node_spacing=60
):
    """
    Generate SVG code to display the nodes and connections of a simple neural net.

    Args:
      neural_net (list of list of float): A list of layers,
        where each layer is a list of node values.
      width (int): Width of the SVG canvas.
      height (int): Height of the SVG canvas.
      node_radius (int): Radius of each node.
      layer_spacing (int): Horizontal spacing between layers.
      node_spacing (int): Vertical spacing between nodes.

    Returns:
      str: SVG code as a string.
    """

    nodes = []
    edges = []
    text = []
    max_layers = len(neural_net)
    max_nodes = max(max(layer.in_features, layer.out_features) for layer in neural_net)
    width = max_layers * layer_spacing + (node_radius + margin_x) * 2
    height = (max_nodes - 1) * node_spacing + (node_radius + margin_y) * 2

    def get_x(layer_idx):
        return  margin_x + node_radius + layer_idx * layer_spacing

    def get_y(layer_size, node_idx):
        return height / 2 + (node_idx - (layer_size - 1) / 2) * node_spacing

    def add_layer(layer_idx, layer_size):
        x = get_x(layer_idx)
        # Add nodes
        for node_idx in range(layer_size):
            y = get_y(layer_size, node_idx)
            nodes.append(f'\n\t<circle cx="{x}" cy="{y}" r="{node_radius}" />')

    # Add connections between nodes
    for layer_idx, layer in enumerate(neural_net):
        weights = layer.weight.data
        layer_in_size = layer.in_features
        layer_out_size = layer.out_features
        x1 = get_x(layer_idx)
        x2 = get_x(layer_idx + 1)

        for idx2, col2 in enumerate(weights):
            y2 = get_y(layer_out_size, idx2)

            for idx1, weight in enumerate(col2):
                opacity = F.sigmoid(weight).item() ** 2
                colour = "blue" if weight.item() > 0 else "red"

                if opacity > 0.01:
                    y1 = get_y(layer_in_size, idx1)
                    edges.append(
                        f'\n\t<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour}" opacity="{opacity:.2f}"/>'
                    )

    # Add nodes for inputs
    for layer_idx, layer in enumerate(neural_net):
        add_layer(layer_idx, layer.in_features)

    label_x1 = get_x(0) - node_radius - 5
    label_x2 = get_x(max_layers) + node_radius + 5
    for label_idx, label in enumerate(labels):
        y = get_y(neural_net[0].in_features, label_idx)
        text.append(f'\n\t<text x="{label_x1}" y="{y}" text-anchor="end">{html.escape(label)}</text>')
        text.append(f'\n\t<text x="{label_x2}" y="{y}">{html.escape(label)}</text>')

    # Add final layer for outputs
    add_layer(max_layers, neural_net[-1].out_features)

    # Wrap elements in SVG tags
    svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'

    svg_code += STYLES

    svg_code += "".join(edges)
    svg_code += "".join(nodes)
    svg_code += "".join(text)
    svg_code += "\n</svg>"

    return svg_code


def write_svg_to_file(svg_code, filename):
    """
    Write SVG code to a file.

    Args:
      svg_code (str): SVG code as a string.
      filename (str): Name of the output file.
    """
    with open(filename, "w") as f:
        f.write(svg_code)
