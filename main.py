import numpy as np
import os
import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn
import GraphClusterers as grCls
import warnings
import random
from math import exp

warnings.simplefilter("ignore", UserWarning)


def generateGraphSequence(n_steps, step_size, generator, filename, **kwargs):
    graphs = [None] * n_steps
    for i in range(n_steps):
        random.seed(i)
        graphs[i] = generator(offset=i * step_size, **kwargs)
    return graphs


def measureClusteringMethod(graphs, clusterer, generationpars):
    clustering_func = clusterer["method"]
    ks = clusterer["ks"]
    filename = "TestData/" + generationpars["filename"] + "_" + clusterer["filename"] + ".txt"
    # Intialisation
    modularities = []
    consistencies = []
    if not os.path.isfile(filename):
        with open(filename, "a") as file:
            file.write(f"#Results of {clustering_func.__name__} on graphs generated using {generationpars['generator'].__name__}\n")
            file.write("#n_graphs, step_size, k_gen, density, k_cluster, modularity, consistency\n")
    with open(filename) as file:
        data = np.genfromtxt(file, dtype=(int, int, float, float, float, float, float), delimiter=",", skip_header=1, names=True)
    mask = [x["n_graphs"]==generationpars["n_steps"] and x["step_size"]==generationpars["step_size"] and x["k_gen"]==generationpars["k_out"] and 
            x["density"]==generationpars["density"] for x in data]
    # Merging the graphs and calculating the updated modularities and differences
    data = data[mask]
    with open(filename, "a") as file:
        for k in ks:
            mod_sum = 0
            consistency_sum = 0
            for row in data:
                if row["k_cluster"]==k:
                    modularities.append(row["modularity"])
                    consistencies.append(row["consistency"])
                    break
            else:
                print(k)
                partitions = clustering_func(graphs, k)
                for i in range(n_steps):
                    modularity = graphs[i].modularity(partitions[i])
                    # print(f"Modularity on G{i}: {modularity}")
                    mod_sum += modularity
                    if i > 0:
                        consistency_sum += grAn.Consistency(partitions[i], partitions[i - 1])
                modularity = mod_sum / n_steps
                print(f"Modularity average = {modularity}")
                modularities.append(modularity)
                consistency = consistency_sum / (n_steps - 1)
                print(f"Consistency average = {consistency}")
                consistencies.append(consistency)
                file.write(f'{generationpars["n_steps"]}, {generationpars["step_size"]}, {generationpars["k_out"]}, {generationpars["density"]}, {k}, {modularity}, {consistency}\n')

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


def ClusterTest(GenerationPars, clusterers, title, filename):
    # Generating the graphs
    graphs = generateGraphSequence(**GenerationPars)
    # Refference point
    mod_sum = 0
    consistency_sum = 0
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
        mods, consists = measureClusteringMethod(
            graphs, clusterer, GenerationPars
        )
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


# Parameters
n_steps = 2**5
n_turns = 4
k_out = 7

if n_steps > 32 * n_turns:
    raise ValueError(f"At most {32*n_turns} steps are allowed, not {n_steps}")
step_size = 32 * n_turns // n_steps

GenerationPars = {
    "n_steps": n_steps,
    "generator": grGen.GirvanNewmanBenchmark,
    "filename": "GNBenchmark",
    "step_size": step_size,
    "k_out": k_out,
    "density": 0.5,
}
clusterers = [
    # {
    #     "method": grCls.clusterVariance,
    #     "ks": [1, 2, 3, 4, 5, 6, 7, 8],
    #     "label": "Variance of optimal solutions",
    #     "filename": "Variance1",
    # },
    # {
    #     "method": grCls.clusterVariance2,
    #     "ks": [1, 2, 3, 4, 5],
    #     "label": "Variance of optimal solutions with depth 2",
    #     "filename": "Variance2",
    # },
    {
        "method": grCls.clusterStacked,
        "ks": [1, 2, 3, 4, 6, 8, 12, 16, 32],
        "label": "Merge-partition",
        "filename": "Stacked",
    },
    {
        "method": grCls.clusterConnected,
        "ks": [
            0.1,
            0.05,
            0.02,
            0.01,
            0.007,
            0.005,
            0.004,
            0.003,
            0.0025,
            0.002,
            0.0015,
            0.001,
            0.0007,
            0.0005,
            0.0002,
            0.0001,
            0,
        ],
        "label": "Connected-partition",
        "filename": "Connected",
    },
    {
        "method": grCls.consistencyLeiden,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden partition",
        "filename": "Consistency1",
    },
    {
        "method": grCls.initialisedConsistencyLeiden,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Initialised consistency Leiden partition",
        "filename": "Consistency2",
    },
]

title = "Girvan and Newman benchmark with two turns in 16 steps"
# filename = "plotGNbenchmark.pdf"
# filename = "plotCreationDestructionBenchmark.pdf"
filename = "test.pdf"

ClusterTest(GenerationPars, clusterers, title, filename)
