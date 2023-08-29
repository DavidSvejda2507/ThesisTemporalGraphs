import igraph as ig
import numpy as np


def GirvanNewmanBenchmark(k_out):
    k_in = 16 - k_out
    p_in = k_in / 31
    p_out = k_out / 96
    pref_matrix = [
        [p_in, p_out, p_out, p_out],
        [p_out, p_in, p_out, p_out],
        [p_out, p_out, p_in, p_out],
        [p_out, p_out, p_out, p_in],
    ]

    G = ig.Graph.SBM(
        n=128,
        pref_matrix=pref_matrix,
        block_sizes=[32, 32, 32, 32],
        directed=False,
        loops=False,
    )

    G.vs["community"] = [0] * 32 + [1] * 32 + [2] * 32 + [3] * 32
    return G


if __name__ == "__main__":
    print("Testing GraphGenerators")
