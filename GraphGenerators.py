import igraph as ig
import numpy as np


def GirvanNewmanBenchmark(k_out, offset=0, density=1):
    k_in = 16 - k_out
    p_in = density * k_in / 31
    p_out = density * k_out / 96
    pref_matrix = [
        [p_in, p_out, p_out, p_out, p_in],
        [p_out, p_in, p_out, p_out, p_out],
        [p_out, p_out, p_in, p_out, p_out],
        [p_out, p_out, p_out, p_in, p_out],
        [p_in, p_out, p_out, p_out, p_in],
    ]
    n_turns = offset // 32
    offset = offset % 32

    G = ig.Graph.SBM(
        n=128,
        pref_matrix=pref_matrix,
        block_sizes=[offset, 32, 32, 32, 32 - offset],
        directed=False,
        loops=False,
    )
    communities = [0] * offset + [1] * 32 + [2] * 32 + [3] * 32 + [0] * (32 - offset)
    G.vs["community"] = [(community + n_turns) % 4 for community in communities]
    return G


def addWeights(w1, w2):
    if w1 is None and w2 is None:
        raise ValueError(f"Tried to add weights {w1} and {w2}")
    if w1 is None:
        return w2
    if w2 is None:
        return w1
    return w1 + w2


def mergeGraphs(Graph1, Graph2):
    if Graph1.is_weighted():
        Graph1.es["weight_1"] = Graph1.es["weight"]
        del Graph1.es["weight"]
    else:
        Graph1.es["weight_1"] = 1
    if "weight_2" in Graph1.es.attributes():
        del Graph1.es["weight_2"]

    if Graph2.is_weighted():
        Graph2.es["weight_2"] = Graph2.es["weight"]
        del Graph2.es["weight"]
    else:
        Graph2.es["weight_2"] = 1
    if "weight_1" in Graph2.es.attributes():
        del Graph2.es["weight_1"]

    # print(Graph1.es.attributes())
    # print(Graph1.vs.attributes())
    # print(Graph1.attributes())
    G = ig.union([Graph1, Graph2])
    # print(f"Graph 1 has {len(Graph1.vs)} vertices and {len(Graph1.es)} edges")
    # print(f"Graph 2 has {len(Graph2.vs)} vertices and {len(Graph2.es)} edges")
    # print(f"Graph G has {len(G.vs)} vertices and {len(G.es)} edges")
    # print(G.vs.attributes())
    # print(G.es.attributes())
    G.es["weight"] = [
        addWeights(weight_1, weight_2)
        for weight_1, weight_2 in zip(G.es["weight_1"], G.es["weight_2"])
    ]
    # print(G.es["weight"])
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


if __name__ == "__main__":
    print("Testing GraphGenerators")

    mergeGraphs(GirvanNewmanBenchmark(6, 0), GirvanNewmanBenchmark(6, 16))
