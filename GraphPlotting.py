import igraph as ig
import matplotlib.pyplot as plt
import numpy as np

import GraphGenerators as grGen
import GraphAnalysis as grAn
from DataStorage import loadData

import warnings
import os



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
        for ax, j in [(ax1, i-1), (ax2, i), (ax3, i+1)]:
            ax.clear()
            # ig.plot(parts[j], ax, layout = layouts[j], mark_groups = True)
            ig.plot(parts[j], ax, layout = layouts[i])
            ax1.title.set_text(f"{j}")
        fig.canvas.draw() 
        fig.canvas.flush_events() 

def PlotGraphseries(graphs, comm):
    global fig, ax1, ax2, ax3, parts, layouts, i
    i=1
    parts = [ig.VertexClustering(graph, graph.vs[comm]) for graph in graphs]
    layouts = [graph.layout() for graph in graphs]
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (18,7))
    for ax, j in [(ax1, i-1), (ax2, i), (ax3, i+1)]:
        # ig.plot(parts[j], ax, layout = layouts[j], mark_groups = True)
        ig.plot(parts[j], ax, layout = layouts[i])
        ax1.title.set_text(f"{j}")

    fig.canvas.mpl_connect('key_press_event', press)
    
    fig.show()

def loadClusteringMethod(clusterer, generationpars):
    ks = clusterer["ks"]
    filename = "TestData/" + generationpars["filename"] + "_" + clusterer["filename"] + ".txt"
    data = loadData(filename)
    mask = [x["n_graphs"]==generationpars["n_steps"] and x["step_size"]==generationpars["step_size"] and x["k_gen"]==generationpars["k_out"] and 
            x["density"]==generationpars["density"] and x["iterations"]==clusterer["iterations"] for x in data]
    data = data[mask]
    
    modularities = []
    consistencies = []
    for k in ks:
        mod_sum = 0
        consistency_sum = 0
        n=0
        for row in data:
            if row["k_cluster"]==k:
                mod_sum += row["modularity"]
                consistency_sum += row["consistency"]
                n += 1
        if n > 0:
            modularities.append(mod_sum/n)
            consistencies.append(consistency_sum/n)
        else:
            warnings.warn(f"""Failed to find entry with k = {k} in {filename} with n_graphs = {generationpars['n_steps']} and
step_size = {generationpars['step_size']} and k_gen = {generationpars['k_out']} and 
density = {generationpars['density']} and iterations = {clusterer['iterations']}""")
            
    # Interpolating between the different sollutions
    previous = None
    mods, consists = [], []
    for modularity, consistency in zip(modularities, consistencies):
        if previous is not None:
            mods.append(min(previous[0], modularity))
            consists.append(min(previous[1], consistency))
        mods.append(modularity)
        consists.append(consistency)
        previous = (modularity, consistency)
    return mods, consists
    
def PlotTestResults(GenerationPars, clusterers, title, filename, seed=0, iterations=2):
    # Generating the graphs
    graphs = grGen.generateGraphSequence(**GenerationPars, seed_offset=seed)
    # Refference point
    mod_sum = 0
    consistency_sum = 0
    n_steps = GenerationPars["n_steps"]
    for i in range(n_steps):
        partition = graphs[i].vs["community"]
        modularity = graphs[i].modularity(partition)
        mod_sum += modularity
        if i > 0:
            consistency_sum += grAn.Consistency(partition, previous_part)
        previous_part = partition
    modularity = mod_sum / n_steps
    consistency = consistency_sum / (n_steps - 1)
    print(f"Refference modularity: {modularity}")

    # Plotting
    fig, ax = plt.subplots(1, 1)
    for clusterer in clusterers:
        clusterer["iterations"] =  iterations
        mods, consists = loadClusteringMethod(clusterer, GenerationPars)
        ax.plot(consists, mods, "o-", label=clusterer["label"], markevery = 2)
    ax.plot(
        consistency,
        modularity,
        "ro",
        label="'True' community structure",
    )
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.1)
    ax.set_ylabel("Average modularity")
    ax.set_xlabel("Average consistency")
    ax.legend()
    ax.set_title(title)
    fig.savefig(filename)
    
if __name__ == "__main__":
    # import GraphGenerators as GrGen
    # graphs = [GrGen.GirvanNewmanBenchmark(6, i, 1) for i in range(5)]
    # PlotGraphseries(graphs, "community")
    # input("Press return to end\n")
    
    import GraphMeasuring as GrMeas
    gen_pars = GrMeas.GenerationPars
    title = f"{gen_pars['filename']} with {gen_pars['n_steps']*gen_pars['step_size']/32} turns in {gen_pars['n_steps']} steps"
    PlotTestResults(gen_pars, GrMeas.plottable_clusterers, title, "test.pdf")