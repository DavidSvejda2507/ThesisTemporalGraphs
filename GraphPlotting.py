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
    global fig, ax1, ax2, ax3, parts, layouts, i
    print(event.key)
    if event.key == 'right': i = min(i+1, len(parts)-2)
    elif event.key == 'left': i = max(i-1, 1)
    elif event.key == 'enter': plt.close(fig)
    if event.key in ['right', 'left']:
        # print(i)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        # ig.plot(parts[i-1], ax1, layout = layouts[i], mark_groups = True)
        # ig.plot(parts[i  ], ax2, layout = layouts[i], mark_groups = True)
        # ig.plot(parts[i+1], ax3, layout = layouts[i], mark_groups = True)
        ig.plot(parts[i-1], ax1, layout = layouts[i])
        ig.plot(parts[i  ], ax2, layout = layouts[i])
        ig.plot(parts[i+1], ax3, layout = layouts[i])
        # print(dir(ax1.title))
        ax1.title.set_text(f"{i-1}")
        ax2.title.set_text(f"{i}")
        ax3.title.set_text(f"{i+1}")
        fig.canvas.draw() 
        fig.canvas.flush_events() 

def PlotGraphseries(graphs, comm):
    global fig, ax1, ax2, ax3, parts, layouts, i
    i=1
    parts = [ig.VertexClustering(graph, graph.vs[comm]) for graph in graphs]
    layouts = [graph.layout() for graph in graphs]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (18,7))
    # ig.plot(parts[i-1], ax1, layout = layouts[i], mark_groups = True)
    # ig.plot(parts[i  ], ax2, layout = layouts[i], mark_groups = True)
    # ig.plot(parts[i+1], ax3, layout = layouts[i], mark_groups = True)
    ig.plot(parts[i-1], ax1, layout = layouts[i])
    ig.plot(parts[i  ], ax2, layout = layouts[i])
    ig.plot(parts[i+1], ax3, layout = layouts[i])
    # print(dir(ax1.title))
    ax1.title.set_text(f"{i-1}")
    ax2.title.set_text(f"{i}")
    ax3.title.set_text(f"{i+1}")

    fig.canvas.mpl_connect('key_press_event', press)
    
    fig.show()
    
    
if __name__ == "__main__":
    import GraphGenerators as GrGen
    graphs = [GrGen.GirvanNewmanBenchmark(6, i, 1) for i in range(5)]
    PlotGraphseries(graphs, "community")
    input("Press return to end\n")