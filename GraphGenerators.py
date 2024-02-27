import igraph as ig
import numpy as np
import random


def GirvanNewmanBenchmark(k_out, offset=0, density=1):
    k_in = 16 - k_out
    p_in = density * k_in / 31
    p_out = density * k_out / 96
    n_turns = offset // 32
    offset = offset % 32

    communities = [(0, offset), (1, 32), (2, 32), (3, 32), (0, 32 - offset)]
    communities = [((comm + n_turns) % 4, length) for comm, length in communities]

    return stochasticBlockFromCummunities(
        communities=communities, p_in=p_in, p_out=p_out
    )


def TestCreationDestruction(k_out, offset, density=1):
    k_in = 16 - k_out
    p_in = density * k_in / 31
    p_out = density * k_out / 96
    n_turns = offset // 32
    offset = offset % 32

    communities = [
        (0, offset),
        (1, 32),
        (2, 32 - offset),
        (3, offset),
        (4, 32),
        (5, 32 - offset),
    ]
    communities = [((comm + n_turns) % 6, length) for comm, length in communities]

    return stochasticBlockFromCummunities(
        communities=communities, p_in=p_in, p_out=p_out
    )


def stochasticBlockFromCummunities(communities, p_in, p_out):
    pref_matrix = [
        [(p_in if com1 == com2 else p_out) for com2, _ in communities]
        for com1, _ in communities
    ]

    block_sizes = [length for _, length in communities]

    G = ig.Graph.SBM(
        n=sum(block_sizes),
        pref_matrix=pref_matrix,
        block_sizes=block_sizes,
        directed=False,
        loops=False,
    )

    comm_vector = []
    for comm, length in communities:
        comm_vector += [comm] * length

    G.vs["community"] = comm_vector
    return G

def MergingSplitting(k_out, offset, density=1):
    k_in = 16 - k_out
    p_in = density * k_in / 31
    p_out = density * k_out / 96
    n_turns = offset // 32
    offset = offset % 32
    
    if n_turns%2==0:
        progress = 1-abs(offset-16)/16
        p_inter = progress*p_in + (1-progress)*p_out
        pref_matrix = [
            [p_in, p_inter, p_out, p_out],
            [p_inter, p_in, p_out, p_out],
            [p_out, p_out, p_in, p_inter],
            [p_out, p_out, p_inter, p_in],
        ]
        communities = [
            (0,32),
            (0 if progress>0.5 else 2,32),
            (4,32),
            (4 if progress>0.5 else 6,32),
        ]
    else:
        progress = abs(offset-16)/16
        p_inter = progress*p_in + (1-progress)*p_out
        pref_matrix = [
            [p_in, p_inter, p_out, p_out, p_out, p_out, p_out, p_out],
            [p_inter, p_in, p_out, p_out, p_out, p_out, p_out, p_out],
            [p_out, p_out, p_in, p_inter, p_out, p_out, p_out, p_out],
            [p_out, p_out, p_inter, p_in, p_out, p_out, p_out, p_out],
            [p_out, p_out, p_out, p_out, p_in, p_inter, p_out, p_out],
            [p_out, p_out, p_out, p_out, p_inter, p_in, p_out, p_out],
            [p_out, p_out, p_out, p_out, p_out, p_out, p_in, p_inter],
            [p_out, p_out, p_out, p_out, p_out, p_out, p_inter, p_in],
        ]
        communities = [
            (0,16),
            (0 if progress>0.5 else 1,16),
            (2,16),
            (2 if progress>0.5 else 3,16),
            (4,16),
            (4 if progress>0.5 else 5,16),
            (6,16),
            (6 if progress>0.5 else 7,16),
        ]
        
    block_sizes = [length for _, length in communities]

    G = ig.Graph.SBM(
        n=sum(block_sizes),
        pref_matrix=pref_matrix,
        block_sizes=block_sizes,
        directed=False,
        loops=False,
    )

    comm_vector = []
    for comm, length in communities:
        comm_vector += [comm] * length

    G.vs["community"] = comm_vector
    print(comm_vector)
    return G


def concatGraphs(graphs, weight):
    """Takes a list of graphs which all have the same vertices
    and combines them into one graph
    with all of the same vertices of consecutive graphs connected by edges
    with the specified weight

    Args:
        graphs (list of iGraph): Graphs to be connected
        weight (double): Weight of the connecting graphs
    """
    n = graphs[0].vcount()
    n_graphs = len(graphs)
    for index, graph in enumerate(graphs):
        if not graph.is_weighted():
            graph.es["weight"] = 1
        if graph.vcount() != n:
            raise ValueError(
                f"Not all graphs have the same number of vertices:\nGraph 0 has {n} vertices\nGraph {index} has {graph.vcount()} vertices"
            )

    G = ig.operators.disjoint_union(graphs)
    edges = [(x, x + n) for x in range(n * (n_graphs - 1))]
    weights = [weight] * (n * (n_graphs - 1))
    G.add_edges(edges, {"weight": weights})
    return G


def generateGraphSequence(seed_offset, n_steps, step_size, generator, filename=None, **kwargs):
    graphs = [None] * n_steps
    for i in range(n_steps):
        random.seed(int(i + seed_offset*n_steps))
        graphs[i] = generator(offset=i * step_size, **kwargs)
    return graphs


if __name__ == "__main__":
    print("Testing GraphGenerators")

    # GirvanNewmanBenchmark(6, 16)
    # GirvanNewmanBenchmark(6, 48)

    # print()
    # TestCreationDestruction(6, 16)

    for i in range(0, 65, 4):
        MergingSplitting(6, i, 1)