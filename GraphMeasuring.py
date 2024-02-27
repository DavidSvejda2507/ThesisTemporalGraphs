import os
import argparse as ap
import numpy as np

import GraphGenerators as grGen
import GraphAnalysis as grAn
import GraphClusterers as grCls
import DataStorage as DS
from math import exp


GenerationPars = [
    {
        "n_steps": 32,
        "generator": grGen.GirvanNewmanBenchmark,
        "filename": "GNBenchmark",
        "step_size": 2,
        "k_out": 7,
        "density": 0.5,
    },
]
clusterers = [
    {
        "method": grCls.clusterVariance,
        "ks": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": "Variance of optimal solutions",
        "filename": "Variance1",
        "iterations": 2
    },
    {
        "method": grCls.clusterVariance2,
        "ks": [1, 2, 3, 4, 5],
        "label": "Variance of optimal solutions with depth 2",
        "filename": "Variance2",
        "iterations": 2
    },
    {
        "method": grCls.clusterStacked,
        "ks": [1, 2, 3, 4, 6, 8, 12, 16, 32],
        "label": "Merge-partition",
        "filename": "Stacked",
        "iterations": 3
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
        "iterations": 5
    },
    {
        "method": grCls.consistencyLeiden,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden partition",
        "filename": "Consistency1-0",
        "iterations": 5
    },
    {
        "method": grCls.initialisedConsistencyLeiden,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Initialised consistency Leiden partition",
        "filename": "Consistency1-1",
        "iterations": 5
    },
]
initialisable_clusterers = [
    {
        "method": grCls.consistencyLeiden2,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2 partition",
        "filename": "Consistency2-0",
        "iterations": 8
    },
]
plottable_clusterers = clusterers + initialisable_clusterers


def measure(filename, line, initialisable = True):
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"file '{filename}' not found")
    orders = np.genfromtxt(filename, dtype=None,
                            delimiter=",", skip_header=line, max_rows=1, autostrip=True, encoding=None,
                            names=["generator", "clusterer", "n_graphs", "step_size", "k_gen", "density", "seed", "k_cluster", "iterations"])
    order = np.reshape(orders, (1,))[0]
    # Order: "#generator, clusterer, n_graphs, step_size, k_gen, density, seed, k_cluster, iterations\n"
    filename = "TestData/" + str(order["generator"]) + "_" + str(order["clusterer"]) + ".txt"
    clustering_func = [x["method"] for x in clusterers if x["filename"]==str(order["clusterer"])][0]
    generator_func = [x["generator"] for x in GenerationPars if x["filename"]==str(order["generator"])][0]
    
    graphs = grGen.generateGraphSequence(order["seed"], order["n_graphs"], order["step_size"], generator_func, k_out = order["k_gen"], density = order["density"])
    if initialisable:
        partitions = clustering_func(graphs, order["k_cluster"], iterations = 1)
        iterations = 1
    else:
        partitions = clustering_func(graphs, order["k_cluster"], iterations = order["iterations"])
        iterations = order["iterations"]
    mod_sum, consistency_sum = grAn.evaluatePartitions(graphs, partitions)
            
    DS.maybeWriteData(filename, clustering_func, generator_func,
                 modularity=mod_sum, consistency=consistency_sum, iterations=iterations,
                 **order[2:-1])
    if initialisable:
        while iterations < order["iterations"]:
            partitions = clustering_func(graphs, order["k_cluster"], iterations = 1, initialisation = "comm")
            iterations += 1
            mod_sum, consistency_sum = grAn.evaluatePartitions(graphs, partitions)
            DS.maybeWriteData(filename, clustering_func, generator_func,
                        modularity=mod_sum, consistency=consistency_sum, iterations=iterations,
                        **order[2:-1])
            

def generateOrders(filename, seeds, initialisable = False):
    with open(filename, "w") as file:
        for gen_par in GenerationPars:
            if initialisable: cluster_list = initialisable_clusterers
            else: cluster_list = clusterers
            for clusterer in cluster_list:
                _filename = "TestData/" + gen_par["filename"] + "_" + clusterer["filename"] + ".txt"
                data = DS.loadData(_filename)
                if data is None: data = []
                mask = [x["n_graphs"]==gen_par["n_steps"] and x["step_size"]==gen_par["step_size"] and x["k_gen"]==gen_par["k_out"] and 
                        x["density"]==gen_par["density"] and x["iterations"]<=clusterer["iterations"] for x in data]
                data = data[mask]
                for k in clusterer["ks"]:
                    mask1 = [x["k_cluster"] == k for x in data]
                    for s in range(seeds):
                        mask2 = [x["seed"] == s for x in data[mask1]]
                        if initialisable:
                            i = clusterer["iterations"]
                            mask3 = [x["iterations"] == i for x in data[mask1][mask2]]
                            if not any(mask3):
                                file.write(F"{gen_par['filename']}, {clusterer['filename']}, {gen_par['n_steps']}, {gen_par['step_size']}, {gen_par['k_out']}, {gen_par['density']}, {s}, {k}, {i}\n")
                        else:
                            for i in range(clusterer["iterations"]):
                                mask3 = [x["iterations"] == i+1 for x in data[mask1][mask2]]
                                if not any(mask3):
                                    file.write(F"{gen_par['filename']}, {clusterer['filename']}, {gen_par['n_steps']}, {gen_par['step_size']}, {gen_par['k_out']}, {gen_par['density']}, {s}, {k}, {i+1}\n")
                                           
    
if __name__ == "__main__":
    # generateOrders("orders_false.txt", 3, False)
    # generateOrders("orders_true.txt", 3, True)
    
    parser = ap.ArgumentParser()
    parser.add_argument("-i", action = "store_true")
    parser.add_argument("fname", type=str)
    parser.add_argument("n", type=int)
    args = parser.parse_args()
    
    measure(args.fname, args.n, args.i)