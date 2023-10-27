import numpy as np
import igraph as ig
import random
from icecream import ic
from math import exp
from queue import SimpleQueue

import GraphGenerators as GrGen
import matplotlib.pyplot as plt

_comm = "leiden_commLeiden"
_refine = "leiden_commLeidenRefinement"
_refineIndex = "leiden_refinementIndex"
_queued = "leiden_queued"
_multiplicity = "leiden_multiplicity"
_degree = "leiden_degree"
_selfEdges = "leiden_selfEdges"
_m = "leiden_m"

theta = 0.02


def leiden(graph, attr, iterations):
    i = 0
    checkWeights(graph)
    communities = initialisePartition(graph, _comm)

    while i < iterations:
        j = 0
        localMove(graph, communities)
        # ic("Local Move")
        communities = cleanCommunities(communities)
        # ic("Clean")

        refine_communities = initialisePartition(graph, _refine)
        converged_inner = refine(graph, communities, refine_communities)
        # ic("Refine Move")
        refine_communities = cleanCommunities(refine_communities)
        # ic("Clean")

        graphs = [graph]

        while not converged_inner:
            _graph = aggregate(graph, refine_communities)
            # ic("Aggregate")
            graphs.append(_graph)
            graph = _graph

            localMove(graph, communities)
            # ic("Local Move inner")
            communities = cleanCommunities(communities)
            # ic("Clean")

            converged_inner = refine(graph, communities, refine_communities)
            # ic("Refine Move inner")
            refine_communities = cleanCommunities(refine_communities)
            # ic("Clean")

            j += 1

        for graph, aggregate_graph in zip(graphs[-2::-1], graphs[:0:-1]):
            deAggregate(graph, aggregate_graph)
        # ic("Deaggregate")
        # ic("")

        graph = graphs[0]
        i += 1

    graph.vs[attr] = graph.vs[_comm]
    del graph.vs[_comm]
    del graph.vs[_refine]
    del graph.vs[_refineIndex]
    del graph.vs[_queued]
    del graph.vs[_degree]
    del graph.vs[_selfEdges]
    del graph[_m]

    renumber(graph, attr)

    return quality(graph, attr)


def quality(graph, comm):
    return graph.modularity(graph.vs[comm])


def commQuality(communities, m):
    modularity = 0
    for comm in communities:
        edgecount, degreesum = communities[comm]
        modularity += edgecount / m - (degreesum / (2 * m)) ** 2
    return modularity


def checkWeights(graph):
    m = 0
    if "weight" in graph.es.attributes():
        for vertex in graph.vs:
            degree = 0
            for neighbor in vertex.neighbors():
                degree += graph[vertex.index, neighbor]
            vertex[_degree] = degree
            m += degree
    else:
        graph.es["weight"] = 1
        graph.vs[_degree] = graph.degree(range(graph.vcount()))
        m = sum(graph.vs[_degree])

    graph.vs[_selfEdges] = 0
    graph.vs[_multiplicity] = 1
    graph[_m] = m / 2


def initialisePartition(graph, attribute):
    communities = {}
    for index, vertex in enumerate(graph.vs):
        vertex[attribute] = index
        communities[index] = (vertex[_multiplicity], 0, vertex[_degree])
    communities[-1] = (
        0,
        0,
        0,
    )  # Moving to community -1 corresponds to moving to a new community
    return communities


def cleanCommunities(communities):
    # ic(communities)
    output = {}
    for key in communities:
        val = communities[key]
        if val[0] != 0:
            output[key] = val
        else:
            if val[1] != 0 or val[2] != 0:
                raise ValueError(
                    f"Community with {val[1]} internal edges and {val[2]} internal degree without internal vertices found"
                )
    output[-1] = (0, 0, 0)
    return output


def makeQueue(count):
    queue = SimpleQueue()
    indices = list(range(count))
    random.shuffle(indices)
    for i in indices:
        queue.put(i)
    return queue


def localMove(graph, communities):
    moved = False

    queue = makeQueue(graph.vcount())
    graph.vs[_queued] = True

    while not queue.empty():
        vertex_id = queue.get()
        degree = graph.vs[vertex_id][_degree]
        current_comm = graph.vs[vertex_id][_comm]
        self_edges = graph.vs[vertex_id][_selfEdges]
        neighbors = graph.neighbors(vertex_id)
        community_edges = {-1: self_edges}
        for vertex in neighbors:
            comm = graph.vs[vertex][_comm]
            community_edges[comm] = (
                community_edges.get(comm, self_edges) + graph[vertex_id, vertex]
            )

        max_dq = 0
        max_comm = current_comm
        cost_of_leaving = calculateDQMinus(
            graph,
            communities,
            current_comm,
            vertex_id,
            community_edges.get(current_comm, self_edges),
            degree,
        )

        for comm in community_edges.keys():
            if comm != current_comm:
                dq = (
                    calculateDQPlus(
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
            #                 message = f"""Moving vertex {vertex_id} from {current_comm} to {max_comm}
            # vertex {vertex_id} has degree {degree}, and {self_edges} selfedges
            # the adjacent communities are {community_edges}""" + test_communities(
            #                     graph, communities, current_comm, max_comm, _comm
            #                 )
            if max_comm == -1:
                ic("split")
                i = 0
                while True:
                    if communities.get(i, (0, 0, 0))[0] == 0:
                        break
                    i += 1
                max_comm = i

            # modularity_old = commQuality(communities, graph[_m])

            graph.vs[vertex_id][_comm] = max_comm
            update_communities(
                communities,
                current_comm,
                max_comm,
                community_edges,
                graph.vs[vertex_id][_multiplicity],
                self_edges,
                degree,
            )

            # modularity_new = commQuality(communities, graph[_m])
            # if abs(modularity_new - modularity_old - max_dq) > 1e-15:
            #     ic(abs(modularity_new - modularity_old - max_dq))

            # test_communities(
            #     graph, communities, current_comm, max_comm, _comm, message
            # )

            for vertex in neighbors:
                if not graph.vs[vertex][_queued]:
                    graph.vs[vertex][_queued] = True
                    queue.put(vertex)

            # cluster = ig.VertexClustering.FromAttribute(graph, _comm)
            # plot = ig.plot(cluster, "test.pdf", layout="circle")
            # plt.close(plot.get_figure())
            # ic(f"{modularity_old}, {modularity_new}")
            # input(f"{max_dq}")

        graph.vs[vertex_id][_queued] = False


def refine(graph, communities, refine_communities):
    converged = True
    for comm in communities:
        kwarg = {_comm + "_eq": comm}
        indices = [v.index for v in graph.vs.select(**kwarg)]
        random.shuffle(indices)

        for vertex_id in indices:
            if refine_communities[graph.vs[vertex_id][_refine]][0] > 1:
                continue
            neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
            degree = graph.vs[vertex_id][_degree]
            current_comm = graph.vs[vertex_id][_refine]
            self_edges = graph.vs[vertex_id][_selfEdges]
            community_edges = {}
            for vertex in neighbors:
                refine_comm = vertex[_refine]
                community_edges[refine_comm] = (
                    community_edges.get(refine_comm, self_edges)
                    + graph[vertex_id, vertex.index]
                )

            candidates = []
            weights = []
            cost_of_leaving = calculateDQMinus(
                graph,
                refine_communities,
                current_comm,
                vertex_id,
                community_edges.get(current_comm, self_edges),
                degree,
            )

            for refine_comm in community_edges.keys():
                dq = (
                    calculateDQPlus(
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
                #                 message = f"""Moving vertex {vertex_id} from {current_comm} to {target}
                # vertex {vertex_id} has degree {degree}, and {self_edges} selfedges
                # the adjacent communities are {community_edges}""" + test_communities(
                #                     graph, refine_communities, current_comm, target, _refine
                #                 )
                # modularity_old = commQuality(refine_communities, graph[_m])

                graph.vs[vertex_id][_refine] = target
                update_communities(
                    refine_communities,
                    current_comm,
                    target,
                    community_edges,
                    graph.vs[vertex_id][_multiplicity],
                    self_edges,
                    degree,
                )
                converged = False

                # modularity_new = commQuality(refine_communities, graph[_m])
                # if abs(modularity_new - modularity_old - dq) > 1e-15:
                #     ic(abs(modularity_new - modularity_old - dq))
                # test_communities(
                #     graph, refine_communities, current_comm, target, _refine, message
                # )
    return converged


def update_communities(
    communities, current, future, community_edges, multiplicity, self_edges, degree
):
    old_vertexcount, old_edgecount, old_degreesum = communities[current]
    communities[current] = (
        old_vertexcount - multiplicity,
        old_edgecount - community_edges.get(current, self_edges),
        old_degreesum - degree,
    )
    old_vertexcount, old_edgecount, old_degreesum = communities[future]
    communities[future] = (
        old_vertexcount + multiplicity,
        old_edgecount + community_edges[future],
        old_degreesum + degree,
    )


def test_communities(graph, communities, old, new, attr, message=""):
    assert old != new
    degree = 0
    count = 0
    stop = False
    vs = graph.vs.select(**{f"{attr}_eq": old})
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            count += graph[vs[i], vs[j]]
        degree += vs[i][_degree]
        count += vs[i][_selfEdges]
    if communities[old] != ic((len(vs), count, degree)):
        stop = True
    old_message = f"Measured: degree = {degree}\t, count = {count}\nStored: degree = {communities[old][2]}\t, count = {communities[old][1]}"
    degree = 0
    count = 0
    vs = graph.vs.select(**{f"{attr}_eq": new})
    test = f"{[v.index for v in vs]}"
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            count += graph[vs[i], vs[j]]
            test += f"\nThe edge from {vs[i].index} to {vs[j].index} has weight {graph[vs[i],vs[j]]}"
        degree += vs[i][_degree]
        count += vs[i][_selfEdges]
    if communities[new] != ic((len(vs), count, degree)):
        stop = True
    new_message = f"Measured: degree = {degree}\t, count = {count}\nStored: degree = {communities[new][2]}\t, count = {communities[new][1]}"

    if stop:
        print(message)
        print("old:")
        print(old_message)
        print("new:")
        print(new_message)
        print(test)
        assert False

    return "old:\n" + old_message + "\nnew:\n" + new_message


"""
modularity of a community =
edge_count/m - (degree_sum/2m)^2
We want to compare having the vertex in the community vs having it in its own community
"""


def calculateDQPlus(graph, communities, comm, vertex, edges, degree):
    vertexcount, edgecount, degreesum = communities[comm]
    dq = edges / graph[_m] - ((2 * degreesum + degree) * degree) / (2 * graph[_m]) ** 2
    return dq


def calculateDQMinus(graph, communities, comm, vertex, edges, degree):
    vertexcount, edgecount, degreesum = communities[comm]
    dq = -edges / graph[_m] + ((2 * degreesum - degree) * degree) / (2 * graph[_m]) ** 2
    return dq


def aggregate(graph, communities):
    partition = ig.VertexClustering.FromAttribute(graph, _refine)
    aggregate_graph = partition.cluster_graph(
        {None: "first", _multiplicity: "sum", _degree: "sum"}, "sum"
    )
    del partition

    _dict = {v[_refine]: v.index for v in aggregate_graph.vs}
    graph.vs[_refineIndex] = [_dict[ref] for ref in graph.vs[_refine]]
    aggregate_graph.vs[_selfEdges] = [
        communities[v[_refine]][1] for v in aggregate_graph.vs
    ]

    return aggregate_graph


def deAggregate(graph, _graph):
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
    ig.config["plotting.backend"] = "matplotlib"
    # graphs = [1, 2, 3, 4, 5, 6]
    # print(graphs[-2::-1])
    # print(graphs[:0:-1])
    # for graph, aggregate_graph in zip(graphs[-2::-1], graphs[:0:-1]):
    #     print(graph, aggregate_graph)

    # g = ig.Graph.Full(4)
    # g.vs["comm"] = [0, 0, 1, 1]
    # g.vs[_degree] = [3, 3, 3, 3]
    # g.es["weight"] = [1, 1, 1, 1, 1, 1]
    # g["m"] = 17
    # part = ig.VertexClustering.FromAttribute(g, "comm")
    # g2 = part.cluster_graph({None: "first", _degree: "sum"}, "sum")
    # print(g2.summary())
    # print(g["m"])
    # print(g2["m"])
    # print(g2.es.attributes())
    # print(g2.es["weight"])
    # print(g2.vs["comm"])
    # print(g2.vs[_degree])

    # ic.disable()
    for i in range(1):
        # graph = ig.Graph.Famous("Cubical")
        # graph = ig.Graph.Famous("zachary")
        # random.seed(49)
        # random.seed(48)
        random.seed(36)
        graph = GrGen.GirvanNewmanBenchmark(7, density=0.5)
        random.seed(36)
        print(leiden(graph, "comm", 10))
        # ic(graph.vs["comm"])
        cluster = ig.VertexClustering.FromAttribute(graph, "comm")
        ig.plot(cluster, "test.pdf")
    # print(graph.vs["comm"])

    # print("###########################################")

    # for i in range(10):
    #     graph = ig.Graph.Famous("zachary")
    #     print(leidenClass(graph, "comm"))
    # # print(graph.vs["comm"])
