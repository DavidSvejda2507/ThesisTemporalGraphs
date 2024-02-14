import igraph as ig
import leidenalg
import numba
import GraphAnalysis as grAn
import LeidenConsistency as LeiCons
import LeidenConsistency2 as LeiCons2


def clusterStacked(graphs, k, iterations = 2):
    """Cluster the graphs by stacking k graphs on top of one another and clustering each stack

    Args:
        graphs ([Graphs]): List with the graphs to be clustered
        k (int): number of graphs per stack

    Returns:
        [[int]]: List of membership vectors
    """
    i = 0
    partitions = []
    while i < len(graphs):
        membership = leidenalg.find_partition_multiplex(
            graphs[i : i + k], leidenalg.ModularityVertexPartition, n_iterations=iterations
        )[0]
        i += k
        if i < len(graphs):
            partitions += [membership] * k
        else:
            partitions += [membership] * (len(graphs) - i + k)

    return partitions


def clusterConnected(graphs, k, iterations = 2):
    """Cluster the graphs by connecting the graphs along the time axis using edges with weight k

    Args:
        graphs ([Graph]): List with the graphs
        k (double): strength of the connections

    Returns:
        [[int]]: List of membership vectors
    """
    for g in graphs:
        g.vs["id"] = list(range(g.vcount()))

    return leidenalg.find_partition_temporal(
        graphs, leidenalg.ModularityVertexPartition, k, n_iterations=iterations
    )[0]


# @numba.jit(nopython=True)
def clusterVariance(graphs, k, iterations = 2):
    clusters = [[None] * k for _ in range(len(graphs))]
    for i, graph_ in enumerate(graphs):
        for j in range(k):
            partition = leidenalg.find_partition(
                graph_, leidenalg.ModularityVertexPartition, n_iterations=iterations, seed=j * 137
            )
            clusters[i][j] = partition.membership
            del partition

    partitions = [None] * len(graphs)
    max = 0
    max_index = None
    for i in range(k):
        for j in range(k):
            consistency = grAn.Consistency(clusters[0][i], clusters[1][j])
            if consistency > max:
                max = consistency
                max_index = i
    partitions[0] = clusters[0][max_index]
    previous = max_index
    for i in range(1, len(graphs)):
        max = 0
        for j in range(k):
            consistency = grAn.Consistency(clusters[i - 1][previous], clusters[i][j])
            if consistency > max:
                max = consistency
                max_index = j
        partitions[i] = clusters[i][max_index]
        previous = max_index

    return partitions


# @numba.jit(nopython=True)
def clusterVariance2(graphs, k, iterations = 2):
    clusters = [[None] * k for _ in range(len(graphs))]
    for i, graph_ in enumerate(graphs):
        for j in range(k):
            partition = leidenalg.find_partition(
                graph_, leidenalg.ModularityVertexPartition, n_iterations=iterations, seed=j * 137
            )
            clusters[i][j] = partition.membership
            del partition

    partitions = [None] * len(graphs)
    max = 0
    max_index = None
    for i in range(k):
        for j in range(k):
            consistency = grAn.Consistency(clusters[0][i], clusters[1][j])
            for m in range(k):
                consistency2 = grAn.Consistency(clusters[1][j], clusters[2][m])
                if consistency + consistency2 > max:
                    max = consistency
                    max_index = i
    partitions[0] = clusters[0][max_index]
    previous = max_index
    for i in range(1, len(graphs) - 1):
        max = 0
        for j in range(k):
            consistency = grAn.Consistency(clusters[i - 1][previous], clusters[i][j])
            for m in range(k):
                consistency2 = grAn.Consistency(clusters[i][j], clusters[i + 1][m])
                if consistency + consistency2 > max:
                    max = consistency
                    max_index = (j, m)
        partitions[i] = clusters[i][max_index[0]]
        previous = max_index[0]
    partitions[-1] = clusters[-1][max_index[1]]

    return partitions

def consistencyLeiden(graphs, k, iterations = 2):
    LeiCons.leiden(graphs, "comm", iterations, k)
    
    return [graph.vs["comm"] for graph in graphs]

def initialisedConsistencyLeiden(graphs, k, iterations = 2):
    membership = leidenalg.find_partition_multiplex(graphs, leidenalg.ModularityVertexPartition)[0]
    for graph in graphs:
        graph.vs["init"] = membership
    LeiCons.leiden(graphs, "comm", iterations, k, "init")

    return [graph.vs["comm"] for graph in graphs]

def consistencyLeiden2(graphs, k, iterations = 2):
    LeiCons2.leiden(graphs, "comm", iterations, k)
    
    return [graph.vs["comm"] for graph in graphs]