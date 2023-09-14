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
