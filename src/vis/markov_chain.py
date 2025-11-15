from collections import defaultdict
import html

STYLES = """
<style>
    .activation { display: none; pointer-events: none; }
    .click-box {fill-opacity: 0;}
    .edge { stroke-width: 2; }
    .node { fill: #ccc; stroke: white; stroke-width: 2; }
    .node-activation { stroke: white; stroke-width: 0.5; }
    text { font-family: sans-serif; font-size: 12px; fill: black; alignment-baseline: middle; text-anchor: middle; }
</style>\n"""

class Node:
    """
    A class representing a node in the Markov chain.
    """

    def __init__(self, name, layer=0):
        self.name = name
        self.edges = {}
        self.layer = layer

    def __repr__(self):
        return f"Node({self.name} layer={self.layer})"


def build_chain(transitions):
    """
    Build a Markov chain from the given transitions.
    """

    frontier = [Node('<>')]
    chain = {}

    while frontier:
        current = frontier.pop(0)
        if current in chain:
            continue
        
        current_name = current.name
        chain[current_name] = current
        next_words = transitions.get(current_name, {})
        next_layer = current.layer + 1

        for next_word, prob in next_words.items():
            if next_word not in chain:
                frontier.append(Node(next_word, next_layer))
            current.edges[next_word] = prob

    return chain


def draw_chain(chain):
    """
    Draw the Markov chain.
    """

    margin_x = 20
    margin_y = 20
    node_radius = 40
    layer_spacing = node_radius * 2 + 100

    layers = defaultdict(list)
    for node in chain.values():
        layers[node.layer].append(node)

    max_layer_size = max(len(layer) for layer in layers.values())

    width = 2 * (margin_x + node_radius) + len(layers) * layer_spacing
    height = 2 * (margin_y + node_radius) + (max_layer_size - 1) * (4 * node_radius)
    svg_code = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">\n'
    svg_code += STYLES

    for layer in sorted(layers.keys()):
        x = layer * layer_spacing + margin_x + node_radius
        layer_size = len(layers[layer])
        y = height / 2 - (layer_size - 1) * node_radius * 2

        for node in layers[layer]:
            svg_code += f'  <circle class="node" cx="{x}" cy="{y}" r="{node_radius}" />\n'
            svg_code += f'  <text x="{x}" y="{y}">{html.escape(node.name)}</text>\n'
            y += 4 * node_radius

            # for next_word, prob in node.edges.items():
            #     next_node = chain[next_word]
            #     next_x = nodes.index(next_node) * (2 * node_radius + margin_x) + margin_x
            #     next_y = next_node.layer * layer_spacing + margin_y
            #     svg_code += f'  <line x1="{x}" y1="{y}" x2="{next_x}" y2="{next_y}" stroke="black" stroke-width="{prob * 2}"/>\n'

    svg_code += '</svg>'
    return svg_code

if __name__ == '__main__':
    transitions = {
        '<>': {'sheep': 1.0},
        'are': {'herbivores': 0.5, 'fluffy': 0.5},
        'herbivores': {'<>': 1.0},
        'fluffy': {'<>': 1.0},
        'sheep': {'are': 1.0},
    }
    transitions = {
        '<>': {'sheep': 1.0},
        'are': {'herbivores': 0.49999142, 'meek': 0.5000086},
        'eat': {'grass': 1.0},
        'grass': {'<>': 1.0},
        'herbivores': {'<>': 1.0},
        'jumping': {'<>': 1.0},
        'like': {'jumping': 1.0},
        'meek': {'<>': 1.0},
        'sheep': {'are': 0.50001776, 'eat': 0.24999303, 'like': 0.24998921},
    }

    chain = build_chain(transitions)
    svg = draw_chain(chain)

    with open('test-chain.svg', "w") as f:
        f.write(svg)

    # for node in chain.values():
    #     print(node)
    #     print(f"{node.name}: {node.edges}")
