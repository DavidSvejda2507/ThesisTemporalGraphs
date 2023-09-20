import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn
import GraphClusterers as grCls
import warnings

warnings.simplefilter("ignore", UserWarning)


def generateGraphSequence(n_steps, step_size, generator, **kwargs):
    graphs = [None] * n_steps
    for i in range(n_steps):
        graphs[i] = generator(offset=i * step_size, **kwargs)
    return graphs


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
            # print(f"Modularity on G{i}: {modularity}")
            mod_sum += modularity
            if i > 0:
                consistency_sum += grAn.Consistency(partitions[i], partitions[i - 1])

        modularities.append(mod_sum / n_steps)
        print(f"Modularity average = {mod_sum/n_steps}")
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


def ClusterTest(GenerationPars, clusterers, title, filename):
    # Generating the graphs
    graphs = generateGraphSequence(**GenerationPars)
    # Refference point
    mod_sum = 0
    consistency_sum = 0
    for i in range(n_steps):
        partition = graphs[i].vs["community"]
        modularity = graphs[i].modularity(partition)
        mod_sum += modularity
        if i > 0:
            consistency_sum += grAn.Consistency(partition, previous_part)
        previous_part = partition
    modularity = mod_sum / n_steps
    consistency = consistency_sum / (n_steps - 1)
    print(f"Refference modularity: {modularity}")

    # Plotting
    fig, ax = plt.subplots(1, 1)
    for clusterer in clusterers:
        mods, consists = measureClusteringMethod(
            graphs, clusterer["method"], clusterer["ks"]
        )
        ax.plot(consists, mods, "o-", label=clusterer["label"])
    ax.plot(
        consistency,
        modularity,
        "ro",
        label="'True' community structure",
    )
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.1)
    ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.1)
    ax.set_ylabel("Average modularity")
    ax.set_xlabel("Average consistency")
    ax.legend()
    ax.set_title(title)
    fig.savefig(filename)


# Parameters
n_steps = 2**5
n_turns = 2
k_out = 7

if n_steps > 32 * n_turns:
    raise ValueError(f"At most {32*n_turns} steps are allowed, not {n_steps}")
step_size = 32 * n_turns // n_steps

GenerationPars = {
    "n_steps": n_steps,
    "step_size": step_size,
    "generator": grGen.GirvanNewmanBenchmark,
    "k_out": k_out,
    "density": 0.5,
}
clusterers = [
    {
        "method": grCls.clusterVariance,
        "ks": [1, 2, 3, 4, 5, 6, 7, 8],
        "label": "Variance of optimal solutions",
    },
    {
        "method": grCls.clusterStacked,
        "ks": [1, 2, 3, 4, 6, 8, 12, 16, 32],
        "label": "Merge-partition",
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
    },
]

title = "Girvan and Newman benchmark with two turns in 16 steps"
filename = "plotGNbenchmark.pdf"
# filename = "plotCreationDestructionBenchmark.pdf"

ClusterTest(GenerationPars, clusterers, title, filename)
