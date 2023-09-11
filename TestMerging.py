import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn
import warnings

warnings.simplefilter("ignore", UserWarning)


def calcDifference(partition, previous):
    return 1 - grAn.calculateAccuracy(partition, previous)[0]


# Parameters
n_steps = 2**4
n_turns = 2
k_out = 6

# Intialisation
modularities = []
differences = []
mod_sum = 0
diff_sum = 0

if n_steps > 32 * n_turns:
    raise ValueError(f"At most {32*n_turns} steps are allowed, not {n_steps}")
step_size = 32 * n_turns // n_steps

# Generating the graphs
graphs = [None] * n_steps
for i in range(n_steps):
    graphs[i] = grGen.GirvanNewmanBenchmark(k_out, i * step_size, 0.5)
    partition = leidenalg.find_partition(
        graphs[i],
        leidenalg.ModularityVertexPartition,
    )
    modularity = graphs[i].modularity(partition)
    print(f"Modularity on G{i}: {modularity}")
    mod_sum += modularity
    if i > 0:
        diff_sum += calcDifference(partition.membership, previous_comms)
    previous_comms = partition.membership
modularities.append(mod_sum / n_steps)
mod_sum = 0
differences.append(diff_sum)
diff_sum = 0

merging_list = [(index, index + 1, graph) for index, graph in enumerate(graphs)]

# Merging the graphs and calculating the updated modularities and differences
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
            mod_sum += modularity
            if j > 0:
                diff_sum += calcDifference(partition.membership, previous_comms)
            previous_comms = partition.membership
        next_list.append((index_low, index_high, graph))

    modularities.append(mod_sum / n_steps)
    mod_sum = 0
    differences.append(diff_sum)
    diff_sum = 0
    merging_list = next_list
modularities.append(0)
differences.append(0)

# Interpolating between the different sollutions
previous = None
mods, diffs = [], []
for modularity, difference in zip(modularities, differences):
    if previous is not None:
        mods.append(min(previous[0], modularity))
        diffs.append(max(previous[1], difference))
    mods.append(modularity)
    diffs.append(difference)
    previous = (modularity, difference)

# Refference point
mod_sum = 0
diff_sum = 0
for i in range(n_steps):
    partition = graphs[i].vs["community"]
    modularity = graphs[i].modularity(partition)
    print(f"Modularity on G{i}: {modularity}")
    mod_sum += modularity
    if i > 0:
        diff_sum += calcDifference(partition, previous_comms)
    previous_comms = partition

# Plotting
fig, ax = plt.subplots(1, 1)
ax.plot(diffs, mods, "bo-", label="Merge-partition")
ax.plot(diff_sum, mod_sum / n_steps, "ro", label="'True' sollution")
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_ylabel("Modularity")
ax.set_xlabel("Average number of community switches")
ax.legend()
fig.savefig("plot.pdf")
