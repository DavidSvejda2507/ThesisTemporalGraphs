import igraph as ig
import Leiden
import LeidenConsistency
import LeidenConsistency2
import leidenalg
import GraphGenerators as GrGen

import matplotlib.pyplot as plt
import numpy as np
import random

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("n", type=int, help="The number of runs of all algorithms")
args = parser.parse_args()

n = args.n
n_iterations = 2
data = np.zeros((100,), dtype=float)

method = n % 10
seed = (n//10) * 100


for i in range(100):
    random.seed(seed + i)
    G = GrGen.GirvanNewmanBenchmark(7, density=1)
    title = "Histogram of modularities on the Grivan & Newman benchmark"
    
    random.seed(seed + i)
    if method == 0:
        data[i] = leidenalg.find_partition(
            G, leidenalg.ModularityVertexPartition, n_iterations=n_iterations,
        ).modularity
    if method == 1:
        data[i] = Leiden.leiden(G, "comm", n_iterations)
    if method == 2:
        data[i] = Leiden.leiden(G, "comm", n_iterations, True)
    if method == 3:
        data[i] = LeidenConsistency.leiden([G], "comm", n_iterations,0)
    if method == 4:
        data[i] = LeidenConsistency2.leiden([G], "comm", n_iterations,0)

filename = f"ValidationData/{method}_{seed}.npy"
np.save(filename, data)
