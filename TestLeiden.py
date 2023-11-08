import igraph as ig
import Leiden
import leidenalg
import GraphGenerators as GrGen

import matplotlib.pyplot as plt
import numpy as np
import random

n_bins = 15
n = 1000
n_iterations = 2
x = [[None] * n for i in range(2)]

for i in range(n):
    if i % 10 == 0:
        print(i)
    random.seed(i)
    G = GrGen.GirvanNewmanBenchmark(7, density=1)
    title = "Histogram of modularities on the Grivan & Newman benchmark"
    # G = ig.Graph.Famous("Zachary")
    # title = "Histogram of modularities on Zachary's karate klub"
    random.seed(i)
    x[1][i] = Leiden.leiden(G, "comm", n_iterations)
    # if x[1][i] < 0.38:
    #     print(i)
    random.seed(i)
    x[0][i] = leidenalg.find_partition(
        G, leidenalg.ModularityVertexPartition, n_iterations=n_iterations
    ).modularity

fig, ax = plt.subplots(nrows=1, ncols=1)

colors = ["red", "lime"]
labels = ["leidenalg", "python implementation"]
ax.hist(x, n_bins, histtype="bar", color=colors, label=labels)
ax.legend(prop={"size": 10})
ax.set_title(title)

fig.savefig("Leiden_comp.pdf")
