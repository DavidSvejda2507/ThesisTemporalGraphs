import igraph as ig
import leidenalg


def clusterStacked(graphs, k):
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
            graphs[i : i + k], leidenalg.ModularityVertexPartition
        )[0]
        i += k
        if i < len(graphs):
            partitions += [membership] * k
        else:
            partitions += [membership] * (len(graphs) - i + k)

    return partitions


def clusterConnected(graphs, k):
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
        graphs, leidenalg.ModularityVertexPartition, k
    )[0]


def clusterVariance(graphs, k):
    import GraphAnalysis as grAn

    clusters = [[None] * k for _ in range(len(graphs))]
    for i, graph_ in enumerate(graphs):
        for j in range(k):
            partition = leidenalg.find_partition(
                graph_, leidenalg.ModularityVertexPartition, seed=j * 137
            )
            clusters[i][j] = partition.membership

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
