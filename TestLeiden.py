import igraph as ig
import Leiden
import leidenalg
import GraphGenerators as GrGen

import matplotlib.pyplot as plt
import numpy as np
import random

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("n_runs", type=int, help="The number of runs of all algorithms")
args = parser.parse_args()

n_bins = 15
n = args.n_runs
n_iterations = 2
x = [[None] * n for i in range(3)]

for i in range(n):
    if i % 10 == 0:
        print(i)
    random.seed(i)
    G = GrGen.GirvanNewmanBenchmark(7, density=1)
    title = "Histogram of modularities on the Grivan & Newman benchmark"
    # G = ig.Graph.Famous("Zachary")
    # title = "Histogram of modularities on Zachary's karate klub"
    random.seed(i)
    x[0][i] = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, n_iterations=n_iterations,
    ).modularity
    random.seed(i)
    x[1][i] = Leiden.leiden(G, "comm", n_iterations)
    random.seed(i)
    x[2][i] = Leiden.leiden(G, "comm", n_iterations, True)

fig, ax = plt.subplots(nrows=1, ncols=1)

colors = ["red", "lime", "blue"]
labels = ["leidenalg", "python implementation", "simplified python implementation"]
ax.hist(x, n_bins, histtype="bar", color=colors, label=labels)
ax.legend(prop={"size": 10})
ax.set_title(title)

fig.savefig("Leiden_comp.pdf")
