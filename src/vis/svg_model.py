import html
import torch
import torch.nn.functional as F

STYLES = """
<style>
    .activation { display: none; pointer-events: none; }
    .click-box {fill-opacity: 0;}
    .edge { stroke-width: 2; }
    .node { fill: #ccc; stroke: white; stroke-width: 2; }
    .node-activation { stroke: white; stroke-width: 0.5; }
    text { font-family: sans-serif; font-size: 12px; fill: black; alignment-baseline: middle; }
</style>\n"""

SCRIPT = """
<script>
	const clickBoxes = document.querySelectorAll('.click-box');
	const activations = document.querySelectorAll('.activation');
	const weights = document.querySelector('.weights');
	let selectedBox = -1;

	const selectBox = (i) => {
		selectedBox = i;
		activations[i].style.display = 'block';
		weights.style.display = 'none';
	};

	clickBoxes.forEach((box, index) => {
		const handleEvent = () => {
			if (selectedBox === -1) {
				selectBox(index);
			} else if (selectedBox === index) {
				// Deselect the already selected box
				activations[index].style.display = 'none';
				weights.style.display = 'block';
				selectedBox = -1;
			} else {
				// Deselect the previously selected box and select the new one
				activations[selectedBox].style.display = 'none';
				selectBox(index);
			}
		};

		box.addEventListener('click', handleEvent);
		box.addEventListener('touchstart', (e) => {
			e.preventDefault(); // Prevents touch from triggering click
			handleEvent();
		});
	});
</script>"""

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


class Node:
    """
    A class representing a node in the neural network.
    """

    def __init__(self, x, y, radius, activation=0):
        self.x = x
        self.y = y
        self.radius = radius
        self.activation = activation
        self.out_edges = []

    def activate(self, activation):
        """
        Set the activation of the node.
        """
        self.activation = activation
        for edge in self.out_edges:
            edge.activation = activation * edge.weight

    def __str__(self):
        if abs(self.activation) < 0.01:
            return f'<circle class="node" cx="{self.x}" cy="{self.y}" r="{self.radius}" />'

        active_radius = self.radius * abs(self.activation)
        colour = "#48f" if self.activation > 0 else "red"

        contents = f"""
            <circle class="node" cx="0" cy="0" r="{self.radius}" />
            <circle class="node-activation" cx="0" cy="0" r="{active_radius:.1f}" fill="{colour}" />
            <text x="0" y="0" text-anchor="middle">{self.activation:.2f}</text>"""

        return wrap_in_group(contents, {"transform": f"translate({self.x} {self.y})"})


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
        self.activation = None

    def get_attributes(self):
        value = self.weight if self.activation is None else self.activation
        value = torch.tanh(value).item()
        opacity = abs(value)
        colour = "blue" if value > 0 else "red"
        return {
            "stroke": colour,
            "opacity": f"{opacity:.2f}"
        }

    def __str__(self):
        attr = self.get_attributes()

        if attr["opacity"] == "0.00":
            return ""

        attr_string = get_attribute_string(attr)
        return f'<line class="edge" x1="{self.x1}" y1="{self.y1}" x2="{self.x2}" y2="{self.y2}" {attr_string}/>'


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
        self.n_layers = len(neural_net)
        self.width = self.n_layers * self.layer_spacing + (self.node_radius + self.margin_x) * 2
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
        # For each set of weights, add the input nodes
        for layer_idx, layer in enumerate(self.neural_net):
            self.add_nodes_in_layer(layer_idx, layer.in_features)

        # Add final output nodes
        self.add_nodes_in_layer(len(self.neural_net), self.neural_net[-1].out_features)

    def add_edges(self):
        for layer_idx, layer in enumerate(self.neural_net):
            weights = layer.weight.data

            for y2, col2 in enumerate(weights):
                node2 = self.nodes[(layer_idx + 1, y2)]

                for y1, weight in enumerate(col2):
                    node1 = self.nodes[(layer_idx, y1)]
                    edge = Edge(node1.x, node1.y, node2.x, node2.y, weight)
                    node1.out_edges.append(edge)
                    self.edges.append(edge)

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

    def add_selection_trigger(self, activations):
        """
        Add a rect to capture the a click event to show the  activation of the node.
        """

        width = self.node_radius * 2 + 60
        height = self.node_radius * 2
        x = self.get_x(0) - width + self.node_radius
        attrs = {'x': x, 'width': width, 'height': height}

        svg_code = ''
        n = len(activations)

        for i in range(n):
            attrs['y'] = self.get_y(n, i) - self.node_radius
            svg_code += f'\n\t<rect id="click-box-{i}" class="click-box" {get_attribute_string(attrs)} />'

        return svg_code

    def generate_svg(self):
        # Wrap elements in SVG tags
        svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">'
        svg_code += STYLES

        svg_code += "".join(f"\n\t{edge}" for edge in self.edges if edge)
        svg_code += "".join(f"\n\t{node}" for node in self.nodes.values())
        svg_code += "".join(self.text)
        svg_code += "\n</svg>"

        return svg_code

    def generate_activation_svg(self, activations):
        """ Given a list of activations, generate the SVG code that allows each to be show. """

        # Wrap elements in SVG tags
        svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}">'
        svg_code += STYLES

        edge_code = "".join(f"\n\t{edge}" for edge in self.edges)
        svg_code += wrap_in_group(edge_code, { 'class': 'weights'})

        svg_code += "".join(f"\n\t{node}" for node in self.nodes.values())
        svg_code += "".join(self.text)

        svg_code += self.add_selection_trigger(activations)

        for idx, activation in enumerate(activations):
            self.activate(activation)
            activation_code = "".join(f"\n\t{edge}" for edge in self.edges if f"{edge}")
            activation_code += "".join(f"\n\t{node}" for node in self.nodes.values())
            attrs = { 'id': f'activation-{idx}', 'class': 'activation'}
            svg_code += wrap_in_group(activation_code, attrs)

        svg_code += SCRIPT
        svg_code += "\n</svg>"

        return svg_code


    def activate(self, activation):
        """
        For a given input, activation, calculate the activation of every node in the network.
        """

        def activate_layer(layer_idx, activation):
            """ Activate each node in the layer """
            for node_idx, node_value in enumerate(activation):
                node_id = (layer_idx, node_idx)
                self.nodes[node_id].activate(node_value.item())

        for layer_idx, layer in enumerate(self.neural_net):
            activation_values = activation if layer_idx == 0 else torch.sigmoid(activation)
            activate_layer(layer_idx, activation_values)
            # Calulate the activation of the next layer
            activation = layer(activation)

        # Show the activation of the final layer after softmax
        activate_layer(self.n_layers, torch.softmax(activation, dim=0))

    def write_to_file(self, filename, activations=None):
        """
        Write SVG code to a file.

        Args:
        svg_code (str): SVG code as a string.
        filename (str): Name of the output file.
        """

        with open(filename, "w") as f:
            if activations is not None:
                f.write(self.generate_activation_svg(activations))
            else:
                f.write(self.generate_svg())

    def __str__(self):
        return self.generate_svg()
