import os
import numpy as np

import GraphGenerators as grGen
import GraphAnalysis as grAn
import GraphClusterers as grCls
from math import exp


GenerationPars = {
    "n_steps": 32,
    "generator": grGen.GirvanNewmanBenchmark,
    "filename": "GNBenchmark",
    "step_size": 2,
    "k_out": 7,
    "density": 0.5,
}
clusterers = [
    {
        "method": grCls.clusterVariance,
        "ks": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": "Variance of optimal solutions",
        "filename": "Variance1",
    },
    {
        "method": grCls.clusterVariance2,
        "ks": [1, 2, 3, 4, 5],
        "label": "Variance of optimal solutions with depth 2",
        "filename": "Variance2",
    },
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
    {
        "method": grCls.consistencyLeiden2,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2 partition",
        "filename": "Consistency2-0",
    },
]


def measure(filename, line):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"file '{filename}' not found")
    with open(filename) as file:
        orders = np.genfromtxt(file, dtype=(str, str, int, int, float, float, int, float, int),
                               delimiter=",", skip_header=1, 
                               names=["generator", "clusterer", "n_graphs", "step_size", "k_gen", "density", "seed", "k_cluster", "iterations"])
    order = orders[line]
    # Order: "#generator, clusterer, n_graphs, step_size, k_gen, density, seed, k_cluster, iterations\n"
    filename = "TestData/" + order["generator"] + "_" + order["clusterer"] + ".txt"
    clustering_func = [x["method"] for x in clusterers if x["filename"]==order["clusterer"]][0]
    generator_func = [x["generator"] for x in GenerationPars if x["filename"]==order["generator"]][0]
    
    graphs = grGen.generateGraphSequence(order["seed"], order["n_graphs"], order["step_size"], generator_func, k_out = order["k_gen"], density = order["density"])
    partition = clustering_func(graphs, order["k_cluster"])
    