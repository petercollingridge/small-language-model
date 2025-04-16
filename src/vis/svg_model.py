import html
import torch.nn.functional as F

STYLES = """
<style>
    circle { fill: lightblue; stroke: white; stroke-width: 2; }
    line { stroke-width: 2; }
    text { font-family: sans-serif; font-size: 12px; fill: black; alignment-baseline: middle; }
</style>\n"""


class Node:
    """
    A class representing a node in the neural network.
    """

    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def __str__(self):
        return f'<circle cx="{self.x}" cy="{self.y}" r="{self.radius}" />'


class Edge:
    """
    A class representing an edge between two nodes in the neural network.
    """

    def __init__(self, x1, y1, x2, y2, weight):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight

    def get_attributes(self):
        opacity = F.sigmoid(self.weight).item() ** 2
        colour = "blue" if self.weight.item() > 0 else "red"
        return {
            "stroke": colour,
            "opacity": f"{opacity:.2f}"
        }

    def __str__(self):
        attr = self.get_attributes()
        attr_string = " ".join(f'{key}="{value}"' for key, value in attr.items())
        return f'<line x1="{self.x1}" y1="{self.y1}" x2="{self.x2}" y2="{self.y2}" {attr_string}/>'


class NeuralNetSVG:
    """
    A class to generate SVG representation of a neural network.
    """

    def __init__(self, neural_net, labels=None):
        self.neural_net = neural_net
        self.labels = labels
        self.nodes = {}
        self.edges = []
        self.text = []

        self.margin_x = 100
        self.margin_y = 10
        self.node_radius = 20
        self.layer_spacing = 150
        self.node_spacing = 60

        max_nodes = max(max(layer.in_features, layer.out_features) for layer in neural_net)
        self.width = len(neural_net) * self.layer_spacing + (self.node_radius + self.margin_x) * 2
        self.height = (max_nodes - 1) * self.node_spacing + (self.node_radius + self.margin_y) * 2

        self.add_nodes()
        self.add_edges()
        self.add_labels()

    def get_x(self, layer_idx): 
        return self.margin_x + self.node_radius + layer_idx * self.layer_spacing

    def get_y(self, layer_size, node_idx):
        return self.height / 2 + (node_idx - (layer_size - 1) / 2) * self.node_spacing

    def add_nodes_in_layer(self, layer_idx, layer_size):
        x = self.get_x(layer_idx)
        # Add nodes
        for node_idx in range(layer_size):
            y = self.get_y(layer_size, node_idx)
            node_id = (layer_idx, node_idx)
            self.nodes[node_id] = Node(x, y, self.node_radius)

    def add_nodes(self):
        # Add nodes for inputs
        for layer_idx, layer in enumerate(self.neural_net):
            self.add_nodes_in_layer(layer_idx, layer.in_features)

        # Add final layer for outputs
        self.add_nodes_in_layer(len(self.neural_net), self.neural_net[-1].out_features)

    def add_edges(self):
        for layer_idx, layer in enumerate(self.neural_net):
            weights = layer.weight.data

            for y2, col2 in enumerate(weights):
                node2 = self.nodes[(layer_idx + 1, y2)]

                for y1, weight in enumerate(col2):
                    node1 = self.nodes[(layer_idx, y1)]
                    self.edges.append(Edge(node1.x, node1.y, node2.x, node2.y, weight))

    def add_labels(self):
        self.add_label_col(0, -self.node_radius - 5, True)
        self.add_label_col(len(self.neural_net), self.node_radius + 5)

    def add_label_col(self, col, offset, align_right = False):
        attr = 'text-anchor="end"' if align_right else ''

        for y, label in enumerate(self.labels):
            node1 = self.nodes[(col, y)]
            x = node1.x + offset
            y = node1.y

            self.text.append(f'\n\t<text x="{x}" y="{y}" {attr}>{html.escape(label)}</text>')

    def generate_svg(self):
        # Wrap elements in SVG tags
        svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">'

        svg_code += STYLES

        svg_code += "".join(f"\n\t{edge}" for edge in self.edges)
        svg_code += "".join(f"\n\t{node}" for node in self.nodes.values())
        svg_code += "".join(self.text)
        svg_code += "\n</svg>"

        return svg_code

    def write_to_file(self, filename):
        """
        Write SVG code to a file.

        Args:
        svg_code (str): SVG code as a string.
        filename (str): Name of the output file.
        """

        with open(filename, "w") as f:
            f.write(self.generate_svg())

    def __str__(self):
        return self.generate_svg()
