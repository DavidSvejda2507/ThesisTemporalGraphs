import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn
import warnings

warnings.simplefilter("ignore", UserWarning)

n_steps = 2**4
k_out = 8

if n_steps > 32:
    raise ValueError(f"At most 32 steps are allowed, not {n_steps}")
step_size = 32 // n_steps

graphs = [None] * n_steps
for i in range(n_steps):
    graphs[i] = grGen.GirvanNewmanBenchmark(k_out, i * step_size)
    partition = leidenalg.find_partition(
        graphs[i],
        leidenalg.ModularityVertexPartition,
    )
    modularity = graphs[i].modularity(partition)
    print(f"Modularity on G{i}: {modularity}")

merging_list = [(index, index + 1, graph) for index, graph in enumerate(graphs)]

while len(merging_list) > 1:
    print("Merging")
    next_list = []
    for i in range(0, len(merging_list), 2):
        graph = grGen.mergeGraphs(merging_list[i][2], merging_list[i + 1][2])
        partition = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
        index_low = merging_list[i][0]
        index_high = merging_list[i + 1][1]
        for j in range(index_low, index_high):
            modularity = graphs[j].modularity(partition.membership)
            print(f"Modularity on G{j}: {modularity}")
        next_list.append((index_low, index_high, graph))
    merging_list = next_list
