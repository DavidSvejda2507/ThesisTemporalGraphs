import igraph as ig
import numpy as np


def GirvanNewmanBenchmark(k_out, offset=0):
    k_in = 16 - k_out
    p_in = k_in / 31
    p_out = k_out / 96
    pref_matrix = [
        [p_in, p_out, p_out, p_out, p_in],
        [p_out, p_in, p_out, p_out, p_out],
        [p_out, p_out, p_in, p_out, p_out],
        [p_out, p_out, p_out, p_in, p_out],
        [p_in, p_out, p_out, p_out, p_in],
    ]
    offset = offset % 32

    G = ig.Graph.SBM(
        n=128,
        pref_matrix=pref_matrix,
        block_sizes=[offset, 32, 32, 32, 32 - offset],
        directed=False,
        loops=False,
    )

    G.vs["community"] = (
        [0] * offset + [1] * 32 + [2] * 32 + [3] * 32 + [0] * (32 - offset)
    )
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
    if not Graph1.is_weighted():
        Graph1.es["weight_1"] = 1
    else:
        Graph1.es["weight_1"] = Graph1.es["weight"]
    del Graph1.es["weight_2"]
    del Graph1.es["weight"]

    if not Graph2.is_weighted():
        Graph2.es["weight_2"] = 1
    else:
        Graph2.es["weight_2"] = Graph2.es["weight"]
    del Graph2.es["weight_1"]
    del Graph2.es["weight"]

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


if __name__ == "__main__":
    print("Testing GraphGenerators")

    mergeGraphs(GirvanNewmanBenchmark(6, 0), GirvanNewmanBenchmark(6, 16))
