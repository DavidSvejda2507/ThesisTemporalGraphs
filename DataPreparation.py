import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

from collections import Counter


    
data = np.genfromtxt("RealTest/email-Eu-core-temporal-Dept4.txt",int, delimiter=" ")

n_graphs = 48

# 48 time boundaries that split the dataset into 48 equaly large sets of edges
boundaries_48 = [872976, 2004256, 2977056, 3767771, 4854489, 5552433, 7352538, 8128873, 9157791, 9862136, 10895080, 12209641, 12988506, 14015462, 14601518, 15411513, 16941288, 17737907, 18501237, 19367271, 20065781, 20739262, 21699934, 23078757, 24386643, 26109120, 27300219, 28089537, 29143952, 30336731, 31032619, 31711293, 32404369, 33188676, 33546827, 36642597, 37256159, 38628268, 38998697, 39577660, 40440856, 41214637, 41745432, 42448502, 43408467, 43996646, 44934588, 46464822]

if False:
    max_t = 45464822
    graphs = [None]*n_graphs
    times = []
    for row in data:
        if row[2] > 60000000:
            continue
        times.append(row[2])
    times.sort()
    print(len(times))
    # for i in range(n_graphs):
    #     print(times[len(times)*i//n_graphs])

    print(np.shape(np.digitize(times, boundaries_48, False)))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(np.digitize(times, boundaries_48, False), 100, histtype="bar")
    # ax.hist(times, histtype="bar", bins=boundaries_48)
    # ax.hist(times, n_graphs, histtype="bar", bins=boundaries_48)
    # ax.legend(prop={"size": 10})
    # title = "Histogram of modularities on the Grivan & Newman benchmark"
    # ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Count")

    fig.savefig("test.pdf")
    plt.close()
    
edge_lists = [None]*n_graphs    
for i in range(n_graphs):
    edge_lists[i] = []

buckets = np.digitize(data[:,2], boundaries_48, False)
print(buckets.shape)
    
for row, bucket in zip(data, buckets):
    if row[2] > 5e7:
        continue
    try:
        edge_lists[bucket].append((min(row[0],row[1]),max(row[0],row[1])))
    except IndexError:
        print(bucket)
        
counter = Counter([(0,1), (0,1), (1,2)])
edges, weights = [], []
for edge, weight in counter.items():
    edges.append(edge)
    weights.append(weight)
graph = ig.Graph(n=3, edges = edges, edge_attrs = {"weight": weights})
print(graph[0,1])

for i in range(n_graphs):
    counter = Counter(edge_lists[i])
    edges, weights = [], []
    for edge, weight in counter.items():
        edges.append(edge)
        weights.append(weight)
    graph = ig.Graph(n=142, edges = edges, edge_attrs = {"weight": weights})
    graph.save(f"RealGraphs/Graph_{i}", "pickle")