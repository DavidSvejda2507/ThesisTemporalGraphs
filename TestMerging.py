import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn
import GraphClusterers as grCls
import warnings

warnings.simplefilter("ignore", UserWarning)


def calcDifference(partition, previous):
    return 1 - grAn.calculateAccuracy(partition, previous)[0]


# Parameters
n_steps = 2**4
n_turns = 2
k_out = 7


if n_steps > 32 * n_turns:
    raise ValueError(f"At most {32*n_turns} steps are allowed, not {n_steps}")
step_size = 32 * n_turns // n_steps


def generateGraphSequence(n_steps, step_size, generator, **kwargs):
    graphs = [None] * n_steps
    for i in range(n_steps):
        graphs[i] = generator(offset=i * step_size, **kwargs)
    return graphs


# Generating the graphs
graphs = generateGraphSequence(
    n_steps, step_size, grGen.GirvanNewmanBenchmark, k_out=k_out, density=0.3
)

ks_stacked = [1, 2, 3, 4, 6, 8, 12, 16, 32]
ks_time_sequence = [
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
]


def measureClusteringMethod(graphs, clustering_func, ks):
    # Intialisation
    modularities = []
    consistencies = []
    mod_sum = 0
    consistency_sum = 0
    # Merging the graphs and calculating the updated modularities and differences
    for k in ks:
        print(k)
        partitions = clustering_func(graphs, k)
        for i in range(n_steps):
            modularity = graphs[i].modularity(partitions[i])
            print(f"Modularity on G{i}: {modularity}")
            mod_sum += modularity
            if i > 0:
                consistency_sum += grAn.Consistency(partitions[i], partitions[i - 1])

        modularities.append(mod_sum / n_steps)
        mod_sum = 0
        consistencies.append(consistency_sum / (n_steps - 1))
        consistency_sum = 0

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


# Refference point
mod_sum = 0
consistency_sum = 0
for i in range(n_steps):
    partition = graphs[i].vs["community"]
    modularity = graphs[i].modularity(partition)
    print(f"Modularity on G{i}: {modularity}")
    mod_sum += modularity
    if i > 0:
        consistency_sum += grAn.Consistency(partition, previous_part)
    previous_part = partition

# Plotting
fig, ax = plt.subplots(1, 1)
mods, consists = measureClusteringMethod(graphs, grCls.clusterStacked, ks_stacked)
ax.plot(consists, mods, "bo-", label="Merge-partition")
mods, consists = measureClusteringMethod(
    graphs, grCls.clusterConnected, ks_time_sequence
)
ax.plot(consists, mods, "go-", label="Connected-partition")
ax.plot(
    consistency_sum / (n_steps - 1),
    mod_sum / n_steps,
    "ro",
    label="'True' community structure",
)
ax.set_ylim(bottom=0)
ax.set_xlim(left=0)
ax.set_ylabel("Average modularity")
ax.set_xlabel("Average consistency")
ax.legend()
ax.set_title("Girvan and Newman benchmark with two turns in 16 steps")
fig.savefig("plot.pdf")
