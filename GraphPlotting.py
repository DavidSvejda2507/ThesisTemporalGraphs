import igraph as ig
import matplotlib.pyplot as plt



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
    
def press(event):
    global fig, ax1, ax2, ax3, parts
    print(event)
    ig.plot(parts[1], ax3, mark_groups=False)

def PlotGraphseries(graphs, comm):
    global fig, ax1, ax2, ax3, parts
    parts = [ig.VertexClustering(graph, graph.vs[comm]) for graph in graphs]
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    ig.plot(parts[0], ax1, mark_groups = True)
    ig.plot(parts[1], ax2, mark_groups = True)
    ig.plot(parts[2], ax3, mark_groups = True)
    fig.canvas.mpl_connect('key_press_event', press)
    
    fig.show()
    
    
if __name__ == "__main__":
    import GraphGenerators as GrGen
    graphs = [GrGen.GirvanNewmanBenchmark(6, i, 1) for i in range(3)]
    PlotGraphseries(graphs, "community")