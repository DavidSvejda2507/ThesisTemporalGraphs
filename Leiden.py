import numpy as np
import igraph as ig
import random
from math import exp
from queue import SimpleQueue

_comm = "leiden_commLeiden"
_refine = "leiden_commLeidenRefinement"
_refineIndex = "leiden_refinementIndex"
_queued = "leiden_queued"

theta = 0.05


def leiden(graph, attr):
    i = 0
    communities = initialisePartition(graph, _comm)

    while i < 10:
        j = 0
        converged_inner = localMove(graph, communities)

        communities = cleanCommunities(communities)
        refine_communities = initialisePartition(graph, _refine)
        refine(graph, communities, refine_communities)

        graphs = [graph]

        while j < 5 and not converged_inner:
            _graph = aggregate(graph)
            graphs.append(_graph)
            graph = _graph

            converged_inner = localMove(graph, communities)

            communities = cleanCommunities(communities)
            refine(graph, communities, refine_communities)

            j += 1

        for graph, aggregate_graph in zip(graphs[-2::-1], graphs[:1:-1]):
            deAggragate(graph, aggregate_graph)

        graph = graphs[0]
        i += 1

    graph.vs[attr] = graph.vs[_comm]
    # del graph.vs[_comm]
    del graph.vs[_refine]
    del graph.vs[_refineIndex]
    del graph.vs[_queued]

    renumber(graph, attr)

    return quality(graph, attr)


def quality(graph, comm):
    return graph.modularity(graph.vs[comm])


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
    queue = SimpleQueue()
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
        indices = [v.index for v in graph.vs.select(**kwarg)]
        random.shuffle(indices)

        for vertex_id in indices:
            if refine_communities[graph.vs[vertex_id][_refine]][0] > 0:
                continue
            neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
            community_edges = {}
            for vertex in neighbors:
                refine_comm = vertex[_refine]
                community_edges[refine_comm] = community_edges.get(refine_comm, 0) + 1

            degree = graph.degree(vertex_id)
            current_comm = graph.vs[vertex_id][_refine]
            candidates = []
            weights = []
            cost_of_leaving = -((degree / (2 * graph.ecount())) ** 2)
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
    current_q = edgecount / graph.ecount() - (0.5 * degreesum / graph.ecount()) ** 2
    new_q = (edgecount + edges) / graph.ecount() - (
        0.5 * (degreesum + degree) / graph.ecount()
    ) ** 2

    dq1 = (
        (edgecount + edges) / graph.ecount()
        - (0.5 * (degreesum + degree) / graph.ecount()) ** 2
        - edgecount / graph.ecount()
        + (0.5 * degreesum / graph.ecount()) ** 2
    )
    dq2 = (
        edges / graph.ecount()
        - ((degreesum + degree)) ** 2 / ((2 * graph.ecount())) ** 2
        + (degreesum) ** 2 / ((2 * graph.ecount())) ** 2
    )
    dq = (
        edges / graph.ecount()
        - ((2 * degreesum + degree) * degree) / (2 * graph.ecount()) ** 2
    )
    if abs(dq1 - new_q + current_q) > 1e-5:
        raise ValueError("difference between difference and dq1")
    if abs(dq1 - dq2) > 1e-5:
        raise ValueError("difference between dq1 and dq2")
    if abs(dq2 - dq) > 1e-5:
        raise ValueError("difference between dq2 and dq")
    return dq


def aggregate(graph):
    partition = ig.VertexClustering.FromAttribute(graph, _refine)
    aggregate_graph = partition.cluster_graph("first", "sum")

    _dict = {v[_refine]: v.index for v in aggregate_graph.vs}
    graph.vs[_refineIndex] = [_dict[ref] for ref in graph.vs[_refine]]

    return aggregate_graph


def deAggragate(graph, _graph):
    graph.vs[_comm] = [_graph.vs[index][_comm] for index in graph.vs[_refineIndex]]


def renumber(graph, comm):
    _dict = {}
    i = 0
    for c in graph.vs[comm]:
        val = _dict.get(c, -1)
        if val == -1:
            _dict[c] = i
            i += 1
    graph.vs[comm] = [_dict[c] for c in graph.vs[comm]]


if __name__ == "__main__":
    graph = ig.Graph.Famous("zachary")
    print(leiden(graph, "comm"))

    print(graph.vs["comm"])
