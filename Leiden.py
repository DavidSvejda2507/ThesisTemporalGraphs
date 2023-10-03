import numpy as np
import igraph as ig
import random
from math import exp
from queue import SimpleQueue

_comm = "_commLeiden"
_refine = "_commLeidenRefinement"
_queued = "_queued"

theta = 0.05


def leiden(graph):
    i = 0

    communities = initialisePartition(graph, _comm)

    while i < 2:
        j = 0
        converged_inner = localMove(_graph, communities)

        communities = cleanCommunities(communities)
        refine_communities = initialisePartition(_graph, _refine)
        refine(_graph, communities, refine_communities)

        _graph = graph

        while j < 5 and not converged_inner:
            converged_inner = localMove(_graph, communities)

            communities = cleanCommunities(communities)
            refine_communities = initialisePartition(_graph, _refine)
            refine(_graph, communities, refine_communities)

            _graph = aggregate(_graph)
            j += 1

        deAggregate(graph, _graph)
        i += 1

    return quality(graph)


def quality(graph):
    return graph.modularity(graph.vs[_comm])


def initialisePartition(graph, attribute):
    communities = {}
    for index, vertex in enumerate(graph.vs):
        vertex[attribute] = index
        communities[index] = (0, graph.degree(index))
    return communities


def cleanCommunities(communities):
    output = {}
    for key in communities:
        val = communities[key]
        if val[1] != 0:
            output[key] = val
        else:
            if val[0] != 0:
                raise ValueError(
                    "Community with internal edges without internal vertices found"
                )
    return output


def localMove(graph, communities):
    queue = SimpleQueue(maxsize=graph.vcount())
    indices = list(range(graph.vcount()))
    random.shuffle(indices)
    for i in indices:
        queue.put(i)
    graph.vs[_queued] = True
    moved = False

    while not queue.empty():
        vertex_id = queue.get()
        neighbors = graph.neighbors(vertex_id)
        community_edges = {}
        for vertex in neighbors:
            comm = graph.vs[vertex][_comm]
            community_edges[comm] = community_edges.get(comm, 0) + 1

        degree = graph.degree(vertex_id)
        current_comm = graph.vs[vertex_id][_comm]
        max_dq = 0
        max_comm = current_comm
        cost_of_leaving = calculateDQ(
            graph,
            communities,
            current_comm,
            vertex_id,
            -community_edges.get(current_comm, 0),
            -degree,
        )
        for comm in community_edges.keys():
            if comm != max_comm:
                dq = (
                    calculateDQ(
                        graph,
                        communities,
                        comm,
                        vertex_id,
                        community_edges[comm],
                        degree,
                    )
                    + cost_of_leaving
                )
                if dq > max_dq:
                    max_dq = dq
                    max_comm = comm

        if max_comm != current_comm:
            graph.vs[vertex_id][_comm] = max_comm
            old_edgecount, old_degreesum = communities[current_comm]
            communities[current_comm] = (
                old_edgecount - community_edges.get(current_comm, 0),
                old_degreesum - degree,
            )
            old_edgecount, old_degreesum = communities[max_comm]
            communities[max_comm] = (
                old_edgecount + community_edges[max_comm],
                old_degreesum + degree,
            )

            for vertex in neighbors:
                if (
                    graph.vs[vertex][_comm] != max_comm
                    and not graph.vs[vertex][_queued]
                ):
                    graph.vs[vertex][_queued] = True
                    queue.put(vertex)

            moved = True

        graph.vs[vertex_id][_queued] = False

    return not moved


def refine(graph, communities, refine_communities):
    for comm in communities:
        kwarg = {_comm + "_eq": comm}
        indices = graph.vs.select(**kwarg)
        random.shuffle(indices)

        for vertex_id in indices:
            if refine_communities[graph.vs[vertex_id][_refine]][0] > 0:
                continue
            neighbors = graph.neighbors(vertex_id).select(**kwarg)
            community_edges = {}
            for vertex in neighbors:
                refine_comm = graph.vs[vertex][_refine]
                community_edges[refine_comm] = community_edges.get(refine_comm, 0) + 1

            degree = graph.degree(vertex_id)
            current_comm = graph.vs[vertex_id][_refine]
            candidates = []
            weights = []
            cost_of_leaving = -((degree / (2 * graph.edgecount)) ** 2)
            for refine_comm in community_edges.keys():
                dq = (
                    calculateDQ(
                        graph,
                        refine_communities,
                        refine_comm,
                        vertex_id,
                        community_edges[refine_comm],
                        degree,
                    )
                    + cost_of_leaving
                )
                if dq > 0:
                    candidates.append(refine_comm)
                    weights.append(exp(dq / theta))

            if len(candidates) > 0:
                target = random.choices(candidates, weights)[0]
                graph.vs[vertex_id][_refine] = target
                old_edgecount, old_degreesum = refine_communities[current_comm]
                refine_communities[current_comm] = (
                    old_edgecount - community_edges.get(current_comm, 0),
                    old_degreesum - degree,
                )
                old_edgecount, old_degreesum = refine_communities[target]
                refine_communities[target] = (
                    old_edgecount + community_edges[target],
                    old_degreesum + degree,
                )


"""
modularity of a community =
edge_count/m - (degree_sum/2m)^2
We want to compare having the vertex in the community vs having it in its own community
"""


def calculateDQ(graph, communities, comm, vertex, edges, degree):
    edgecount, degreesum = communities[comm]
    current_q = (
        edgecount / graph.edgecount() - (0.5 * degreesum / graph.edgecount()) ** 2
    )
    new_q = (edgecount + edges) / graph.edgecount() - (
        0.5 * (degreesum + degree) / graph.edgecount()
    ) ** 2

    dq1 = (
        (edgecount + edges) / graph.edgecount()
        - (0.5 * (degreesum + degree) / graph.edgecount()) ** 2
        - edgecount / graph.edgecount()
        + (0.5 * degreesum / graph.edgecount()) ** 2
    )
    dq2 = (
        edges / graph.edgecount()
        - ((degreesum + degree)) ** 2 / ((2 * graph.edgecount())) ** 2
        + (degreesum) ** 2 / ((2 * graph.edgecount())) ** 2
    )
    dq = (
        edges / graph.edgecount()
        - ((2 * degreesum + degree) * degree) / (2 * graph.edgecount()) ** 2
    )
    if abs(dq1 - new_q + current_q) > 1e-5:
        raise ValueError("difference between difference and dq1")
    if abs(dq1 - dq2) > 1e-5:
        raise ValueError("difference between dq1 and dq2")
    if abs(dq2 - dq) > 1e-5:
        raise ValueError("difference between dq2 and dq")
    return dq


if __name__ == "__main__":
    graph = ig.Graph.Full(4)
    graph.vs["comm"] = [0, 0, 1, 1]
    graph.vs["test"] = 0
    partition = ig.VertexClustering.FromAttribute(graph, "comm")
    subgraph = partition.subgraph(0)
    print(graph.vs["test"])
    print(subgraph.vs["test"])
    subgraph.vs["test"] = 1
    print(graph.vs["test"])
    print(subgraph.vs["test"])
