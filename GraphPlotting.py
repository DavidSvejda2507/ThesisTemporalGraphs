import igraph as ig
import matplotlib as plt


def defaultVisualStyle(layout=None):
    visual_style = {}
    visual_style["vertex_size"] = 0.2
    visual_style["edge_width"] = 1
    if layout is not None:
        visual_style["layout"] = layout
    visual_style["bbox"] = (600, 400)
    visual_style["margin"] = 20
    # visual_style["layout"] = "kk"
    return visual_style


# https://ryanwingate.com/visualization/matplotlib/list-of-matplotlib-colors/
def colorMap(inputs=None):
    if inputs is None:
        return ig.drawing.colors.ClusterColoringPalette(20)
    else:
        return {
            key: col
            for key, col in zip(
                inputs, ig.drawing.colors.ClusterColoringPalette(len(inputs))
            )
        }


def shapeMap(inputs=None):
    if inputs is None:
        return {0: "circle", 1: "rectangle", 2: "triangle", 3: "triangle-down"}
    else:
        raise NotImplementedError()
