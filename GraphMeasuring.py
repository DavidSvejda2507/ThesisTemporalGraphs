import os
import argparse as ap
import numpy as np

import GraphGenerators as grGen
import GraphAnalysis as grAn
import GraphClusterers as grCls
import DataStorage as DS
from math import exp
import warnings


GenerationPars = [
    {
        "n_steps": 32,
        "generator": grGen.GirvanNewmanBenchmark,
        "filename": "GNBenchmark",
        "step_size": 2,
        "k_out": 7,
        "density": 0.5,
    },
    {
        "n_steps": 32,
        "generator": grGen.TestCreationDestruction,
        "filename": "CreateDestructt",
        "step_size": 2,
        "k_out": 7,
        "density": 0.5,
    },
    {
        "n_steps": 32,
        "generator": grGen.MergingSplitting,
        "filename": "MergeSplit",
        "step_size": 2,
        "k_out": 7,
        "density": 0.5,
    },
]
clusterers = [
    # {
    #     "method": grCls.clusterVariance,
    #     "ks": [1, 2, 3, 4, 5, 6, 7, 8],
    #     "label": "Variance of optimal solutions",
    #     "filename": "Variance1",
    #     "iterations": 2
    # },
    # {
    #     "method": grCls.clusterVariance2,
    #     "ks": [1, 2, 3, 4, 5],
    #     "label": "Variance of optimal solutions with depth 2",
    #     "filename": "Variance2",
    #     "iterations": 2
    # },
    # {
    #     "method": grCls.clusterStacked,
    #     "ks": [1, 2, 3, 4, 6, 8, 12, 16, 32],
    #     "label": "Merge-partition",
    #     "filename": "Stacked",
    #     "iterations": 3
    # },
    {
        "method": grCls.clusterConnected,
        "ks": [
            0.1,
            0.075,
            0.05,
            0.035,
            0.02,
            0.015,
            0.01,
            0.0085,
            0.007,
            0.006,
            0.005,
            0.0045,
            0.004,
            0.0035,
            0.003,
            0.00275,
            0.0025,
            0.00225,
            0.002,
            0.00175,
            0.0015,
            0.00125,
            0.001,
            0.00085,
            0.0007,
            0.0005,
            0.0005,
            0.0002,
            0.0001,
            0,
        ],
        "label": "Temporal Leiden",
        "filename": "TemporalLeiden",
        "iterations": 8
    },
]
initialisable_clusterers = [
    {
        "method": grCls.consistencyLeiden,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden a",
        "filename": "Consistency1-0",
        "iterations": 100
    },
    {
        "method": grCls.consistencyLeiden3,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden b",
        "filename": "Consistency3-0",
        "iterations": 8
    },
    {
        "method": grCls.initialisedConsistencyLeiden,
        "ks": [0]+[exp(i/6) for i in range(-20, 12)],
        "label": "Initialised consistency Leiden a",
        "filename": "Consistency1-1",
        "iterations": 8
    },
    {
        "method": grCls.initialisedConsistencyLeiden3,
        "ks": [0]+[exp(i/6) for i in range(-20, 12)],
        "label": "Initialised consistency Leiden b",
        "filename": "Consistency3-1",
        "iterations": 8
    },
    {
        "method": grCls.consistencyLeiden2,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2a",
        "filename": "Consistency2-0",
        "iterations": 8
    },
    {
        "method": grCls.consistencyLeiden4,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2b",
        "filename": "Consistency4-0",
        "iterations": 8
    },
    {
        "method": grCls.initialisedConsistencyLeiden2,
        "ks": [0]+[exp(i/6) for i in range(-20, 12)],
        "label": "Initialised consistency Leiden a",
        "filename": "Consistency2-1",
        "iterations": 8
    },
    {
        "method": grCls.initialisedConsistencyLeiden4,
        "ks": [0]+[exp(i/6) for i in range(-20, 12)],
        "label": "Initialised consistency Leiden b",
        "filename": "Consistency4-1",
        "iterations": 8
    },
    {
        "method": grCls.consistencyLeiden5,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2c",
        "filename": "Consistency2-2",
        "iterations": 8
    },
    {
        "method": grCls.consistencyLeiden6,
        "ks": [0]+[exp(i/3) for i in range(-10, 6)],
        "label": "Consistency Leiden 2d",
        "filename": "Consistency2-3",
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
    if initialisable:
        iterations = 1
        clustering_func = [x["method"] for x in initialisable_clusterers if x["filename"]==str(order["clusterer"])][0]
    else:
        iterations = order["iterations"]
        clustering_func = [x["method"] for x in clusterers if x["filename"]==str(order["clusterer"])][0]
    generator_func = [x["generator"] for x in GenerationPars if x["filename"]==str(order["generator"])][0]
    
    graphs = grGen.generateGraphSequence(order["seed"], order["n_graphs"], order["step_size"], generator_func, k_out = order["k_gen"], density = order["density"])
    partitions = clustering_func(graphs, order["k_cluster"], iterations = iterations)
    mod_sum, consistency_sum = grAn.evaluatePartitions(graphs, partitions)
    order_dict = dict(zip(["n_graphs", "step_size", "k_gen", "density", "seed", "k_cluster"], list(order)[2:-1]))
            
    DS.maybeWriteData(filename, clustering_func, generator_func,
                 modularity=mod_sum, consistency=consistency_sum, iterations=iterations,
                 **order_dict)
    if initialisable:
        while iterations < order["iterations"]:
            partitions = clustering_func(graphs, order["k_cluster"], iterations = 1, initialisation = "comm")
            iterations += 1
            mod_sum, consistency_sum = grAn.evaluatePartitions(graphs, partitions)
            DS.maybeWriteData(filename, clustering_func, generator_func,
                        modularity=mod_sum, consistency=consistency_sum, iterations=iterations,
                        **order_dict)
            

def generateOrders(filename, seeds, initialisable = False):
    count = 0
    with open(filename, "w") as file:
        for gen_par in GenerationPars:
            if initialisable: cluster_list = initialisable_clusterers
            else: cluster_list = clusterers
            for clusterer in cluster_list:
                _filename = "TestData/" + gen_par["filename"] + "_" + clusterer["filename"] + ".txt"
                data = DS.loadData(_filename)
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
                                count += 1
                        else:
                            for i in range(clusterer["iterations"]):
                                mask3 = [x["iterations"] == i+1 for x in data[mask1][mask2]]
                                if not any(mask3):
                                    file.write(F"{gen_par['filename']}, {clusterer['filename']}, {gen_par['n_steps']}, {gen_par['step_size']}, {gen_par['k_out']}, {gen_par['density']}, {s}, {k}, {i+1}\n")
                                    count += 1
    return count

def makeSbatch(initialisable, count):
    if initialisable:
        with open("measure2.sbatch", "w") as file:
            file.write(f"""#!/bin/bash

#SBATCH --job-name="order_true"
#SBATCH --array=0-{count-1}
#SBATCH --partition=defq
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

python3 GraphMeasuring.py c -i orders_true.txt $SLURM_ARRAY_TASK_ID

exit 0
""")
    else:
        with open("measure1.sbatch", "w") as file:
            file.write(f"""#!/bin/bash

#SBATCH --job-name="order_false"
#SBATCH --array=0-{count-1}
#SBATCH --partition=defq
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=%j-%x-stdout.txt
#SBATCH --error=%j-%x-stderr.txt

python3 GraphMeasuring.py c orders_false.txt $SLURM_ARRAY_TASK_ID

exit 0""")
                                           
    
if __name__ == "__main__":
    
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    generate = subparsers.add_parser("g", help="Generate missing orders for all pairs of generators and clusterers")
    generate.add_argument("n", type=int, help="Number of seeds generated for each pair generator and clusterer", default=3)
    generate.set_defaults(gen = True)
    
    calculate = subparsers.add_parser("c", help="Perform calculations based on a file of orders")
    calculate.add_argument("-i", action = "store_true", help="Use iterative clusterers")
    calculate.add_argument("fname", type=str, help="filename containing the oders")
    calculate.add_argument("n", type=int, help="Linenumber of the order to be measured")
    calculate.set_defaults(gen = False)
    
    args = parser.parse_args()
    
    warnings.simplefilter("ignore", UserWarning, 55)
    
    if args.gen:
        count = generateOrders("orders_false.txt", args.n, False)
        makeSbatch(False, count)
        count = generateOrders("orders_true.txt", args.n, True)
        makeSbatch(True, count)
    else:
        measure(args.fname, args.n, args.i)
        
    # gp = GenerationPars[2]
    # grGen.generateGraphSequence(0, **gp)
    
