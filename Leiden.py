import numpy as np
import igraph as ig
import random
from icecream import ic
from math import exp
from queue import SimpleQueue

_comm = "leiden_commLeiden"
_refine = "leiden_commLeidenRefinement"
_refineIndex = "leiden_refinementIndex"
_queued = "leiden_queued"
_degree = "leiden_degree"
_selfEdges = "leiden_selfEdges"

theta = 0.02


class LeidenAlg:
    _comm = "leiden_commLeiden"
    _refine = "leiden_commLeidenRefinement"
    _refineIndex = "leiden_refinementIndex"
    _queued = "leiden_queued"
    _degree = "leiden_degree"
    communities = {}
    converged = False

    def __init__(self, graphs, gamma=1, theta=0.05):
        self.graph_stack = [graphs]
        self.gamma = gamma
        self.theta = theta

    def initialisePartition(self, attr):
        communities = {}
        i = 0
        for graph in self.graph_stack[-1]:
            for vertex in graph.vs:
                vertex[attr] = i
                communities[i] = (0, graph.degree(vertex))
                i += 1
        self.communities[attr] = communities
        return self

    def cleanCommunities(self, attr):
        output = {}
        for key in self.communities[attr]:
            val = self.communities[attr][key]
            if val[1] != 0:
                output[key] = val
            else:
                if val[0] != 0:
                    raise ValueError(
                        "Community with internal edges without internal vertices found"
                    )
        self.communities[attr] = output
        return self

    def calculateDQ(self, attr, comm, vertex, edges, degree):
        edgecount, degreesum = self.communities[attr][comm]
        dq = (
            edges / self.graphs[0].ecount()
            - ((2 * degreesum + degree) * degree) / (2 * self.graphs[0].ecount()) ** 2
        )
        return dq

    def checkWeights(self):
        for graph in self.graph_stack[0]:
            if "weight" not in graph.es.attributes():
                graph.es["weight"] = 1
                graph.vs[_degree] = graph.degree(range(graph.vcount()))
            else:
                for vertex in graph.vs:
                    degree = 0
                    for neighbor in vertex.neighbors():
                        degree += graph[vertex.index, neighbor]
                    vertex[_degree] = degree
        return self

    def aggregate(self):
        partition = ig.VertexClustering.FromAttribute(self.graphs[0], _refine)
        self.subgraphs[0] = partition.cluster_graph(
            {None: "first", _degree: "sum"}, "sum"
        )
        del partition

        _dict = {v[_refine]: v.index for v in self.subgraphs[0].vs}
        self.graphs[0].vs[_refineIndex] = [
            _dict[ref] for ref in self.graphs[0].vs[_refine]
        ]
        return self

    def deAggragate(self):
        self.graphs[0].vs[_comm] = [
            self.subgraphs[0].vs[index][_comm]
            for index in self.graphs[0].vs[_refineIndex]
        ]
        return self

    def renumber(self, attr):
        _dict = {}
        i = 0
        for c in self.graphs[0].vs[attr]:
            val = _dict.get(c, -1)
            if val == -1:
                _dict[c] = i
                i += 1
        self.graphs[0].vs[attr] = [_dict[c] for c in self.graphs[0].vs[attr]]
        return self

    def localMove(self, aggregated, attr):
        if aggregated:
            graphs = self.subgraphs
        else:
            graphs = self.graphs
        communities = self.communities[attr]
        queue = SimpleQueue()
        indices = list(range(self.graphs[0].vcount()))
        random.shuffle(indices)
        for i in indices:
            queue.put(i)
        graphs[0].vs[_queued] = True

        while not queue.empty():
            vertex_id = queue.get()
            neighbors = graphs[0].neighbors(vertex_id)
            community_edges = {}
            for vertex in neighbors:
                comm = graphs[0].vs[vertex][_comm]
                community_edges[comm] = community_edges.get(comm, 0) + 1

            degree = graphs[0].degree(vertex_id)
            current_comm = graphs[0].vs[vertex_id][_comm]
            max_dq = 0
            max_comm = current_comm
            cost_of_leaving = self.calculateDQ(
                attr,
                current_comm,
                vertex_id,
                -community_edges.get(current_comm, 0),
                -degree,
            )
            for comm in community_edges.keys():
                if comm != current_comm:
                    dq = (
                        self.calculateDQ(
                            attr,
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
                graphs[0].vs[vertex_id][_comm] = max_comm
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
                        graphs[0].vs[vertex][_comm] != max_comm
                        and not graphs[0].vs[vertex][_queued]
                    ):
                        graphs[0].vs[vertex][_queued] = True
                        queue.put(vertex)

            graphs[0].vs[vertex_id][_queued] = False
        return self

    def refine(self, attr, refine_attr):
        self.converged = True
        communities = self.communities[attr]
        refine_communities = self.communities[refine_attr]
        for comm in communities:
            kwarg = {_comm + "_eq": comm}
            indices = [v.index for v in self.graphs[0].vs.select(**kwarg)]
            random.shuffle(indices)

            for vertex_id in indices:
                if refine_communities[self.graphs[0].vs[vertex_id][_refine]][0] > 0:
                    continue
                neighbors = (
                    self.graphs[0]
                    .vs[self.graphs[0].neighbors(vertex_id)]
                    .select(**kwarg)
                )
                community_edges = {}
                for vertex in neighbors:
                    refine_comm = vertex[_refine]
                    community_edges[refine_comm] = (
                        community_edges.get(refine_comm, 0) + 1
                    )

                degree = self.graphs[0].degree(vertex_id)
                current_comm = self.graphs[0].vs[vertex_id][_refine]
                candidates = []
                weights = []
                cost_of_leaving = -((degree / (2 * self.graphs[0].ecount())) ** 2)
                for refine_comm in community_edges.keys():
                    dq = (
                        self.calculateDQ(
                            refine_attr,
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
                    self.graphs[0].vs[vertex_id][_refine] = target
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
                    self.converged = False
        return self


def leidenClass(graph, attr):
    alg = LeidenAlg([graph])

    i = 0
    alg.checkWeights.initialisePartition(_comm)
    while i < 10:
        j = 0
        # fmt: off
        alg.localMove(False, _comm)\
            .cleanCommunities(_comm)\
            .initialisePartition(_refine)\
            .refine(_comm, _refine)
        # fmt: on
        while j < 5 and not alg.converged:
            # fmt: off
            alg.aggregate()\
                .localMove(True, _comm)\
                .cleanCommunities(_comm)\
                .refine(_comm, _refine)\
                .deAggragate()
            # fmt: on
            j += 1

        i += 1

    graph.vs[attr] = graph.vs[_comm]
    del graph.vs[_comm]
    del graph.vs[_refine]
    del graph.vs[_refineIndex]
    del graph.vs[_queued]

    alg.renumber(attr)
    del alg

    return quality(graph, attr)


def leiden(graph, attr):
    i = 0
    checkWeights(graph)
    communities = initialisePartition(graph, _comm)

    while i < 10:
        j = 0
        converged_inner = localMove(graph, communities)
        # ic("Local Move")
        communities = cleanCommunities(communities)
        # ic("Clean")

        refine_communities = initialisePartition(graph, _refine)
        refine(graph, communities, refine_communities)
        # ic("Refine Move")
        refine_communities = cleanCommunities(refine_communities)
        # ic("Clean")

        graphs = [graph]

        while j < 5 and not converged_inner:
            _graph = aggregate(graph, refine_communities)
            # ic("Aggregate")
            graphs.append(_graph)
            graph = _graph

            converged_inner = localMove(graph, communities)
            # ic("Local Move inner")
            communities = cleanCommunities(communities)
            # ic("Clean")

            refine(graph, communities, refine_communities)
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

    renumber(graph, attr)

    return quality(graph, attr)


def quality(graph, comm):
    return graph.modularity(graph.vs[comm])


def checkWeights(graph):
    if "weight" in graph.es.attributes():
        for vertex in graph.vs:
            degree = 0
            for neighbor in vertex.neighbors():
                degree += graph[vertex.index, neighbor]
            vertex[_degree] = degree
    else:
        graph.es["weight"] = 1
        graph.vs[_degree] = graph.degree(range(graph.vcount()))

    graph.vs[_selfEdges] = 0


def initialisePartition(graph, attribute):
    communities = {}
    for index, vertex in enumerate(graph.vs):
        vertex[attribute] = index
        communities[index] = (0, vertex[_degree])
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
                    f"Community with {val[0]} internal edges without internal vertices found"
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
            community_edges[comm] = (
                community_edges.get(comm, 0) + graph[vertex_id, vertex]
            )

        degree = graph.vs[vertex_id][_degree]
        current_comm = graph.vs[vertex_id][_comm]
        self_edges = graph.vs[vertex_id][_selfEdges]
        max_dq = 0
        max_comm = current_comm
        cost_of_leaving = calculateDQ(
            graph,
            communities,
            current_comm,
            vertex_id,
            -community_edges.get(current_comm, 0) - self_edges,
            -degree,
        )
        for comm in community_edges.keys():
            if comm != current_comm:
                dq = (
                    calculateDQ(
                        graph,
                        communities,
                        comm,
                        vertex_id,
                        community_edges[comm] + self_edges,
                        degree,
                    )
                    + cost_of_leaving
                )
                if dq > max_dq:
                    max_dq = dq
                    max_comm = comm
        if max_comm != current_comm:
            message = f"""Moving vertex {vertex_id} from {current_comm} to {max_comm}
vertex {vertex_id} has degree {degree}, and {self_edges} selfedges
the adjacent communities are {community_edges}
            """ + test_communities(
                graph, communities, current_comm, max_comm, _comm
            )
            graph.vs[vertex_id][_comm] = max_comm
            old_edgecount, old_degreesum = communities[current_comm]
            communities[current_comm] = (
                old_edgecount - community_edges.get(current_comm, 0) - self_edges,
                old_degreesum - degree,
            )
            old_edgecount, old_degreesum = communities[max_comm]
            communities[max_comm] = (
                old_edgecount + community_edges[max_comm] + self_edges,
                old_degreesum + degree,
            )
            test_communities(graph, communities, current_comm, max_comm, _comm, message)

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
                community_edges[refine_comm] = (
                    community_edges.get(refine_comm, 0) + graph[vertex_id, vertex.index]
                )

            degree = graph.vs[vertex_id][_degree]
            current_comm = graph.vs[vertex_id][_refine]
            self_edges = graph.vs[vertex_id][_selfEdges]
            candidates = []
            weights = []
            cost_of_leaving = (degree / (2 * graph.ecount())) ** 2

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
                message = f"""Moving vertex {vertex_id} from {current_comm} to {target}
vertex {vertex_id} has degree {degree}, and {self_edges} selfedges
the adjacent communities are {community_edges}
                """ + test_communities(
                    graph, refine_communities, current_comm, target, _refine
                )
                graph.vs[vertex_id][_refine] = target
                old_edgecount, old_degreesum = refine_communities[current_comm]
                refine_communities[current_comm] = (
                    old_edgecount - community_edges.get(current_comm, 0) - self_edges,
                    old_degreesum - degree,
                )
                old_edgecount, old_degreesum = refine_communities[target]
                refine_communities[target] = (
                    old_edgecount + community_edges[target] + self_edges,
                    old_degreesum + degree,
                )
                test_communities(
                    graph, refine_communities, current_comm, target, _refine, message
                )
            # input()


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
    if communities[old] != (count, degree):
        stop = True
    old_message = f"Measured: degree = {degree}\t, count = {count}\nStored: degree = {communities[old][1]}\t, count = {communities[old][0]}"
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
    if communities[new] != (count, degree):
        stop = True
    new_message = f"Measured: degree = {degree}\t, count = {count}\nStored: degree = {communities[new][1]}\t, count = {communities[new][0]}"

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


def calculateDQ(graph, communities, comm, vertex, edges, degree):
    edgecount, degreesum = communities[comm]
    dq = (
        edges / graph.ecount()
        - ((2 * degreesum + degree) * degree) / (2 * graph.ecount()) ** 2
    )
    return dq


def aggregate(graph, communities):
    partition = ig.VertexClustering.FromAttribute(graph, _refine)
    aggregate_graph = partition.cluster_graph({None: "first", _degree: "sum"}, "sum")
    del partition

    _dict = {v[_refine]: v.index for v in aggregate_graph.vs}
    graph.vs[_refineIndex] = [_dict[ref] for ref in graph.vs[_refine]]
    aggregate_graph.vs[_selfEdges] = [
        communities[v[_refine]][0] for v in aggregate_graph.vs
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
    # graphs = [1, 2, 3, 4, 5, 6]
    # print(graphs[-2::-1])
    # print(graphs[:0:-1])
    # for graph, aggregate_graph in zip(graphs[-2::-1], graphs[:0:-1]):
    #     print(graph, aggregate_graph)

    # g = ig.Graph.Full(4)
    # g.vs["comm"] = [0, 0, 1, 1]
    # g.vs[_degree] = [3, 3, 3, 3]
    # g.es["weight"] = [1, 1, 1, 1, 1, 1]
    # part = ig.VertexClustering.FromAttribute(g, "comm")
    # g2 = part.cluster_graph({None: "first", _degree: "sum"}, "sum")
    # print(g2.summary())
    # print(g2.es.attributes())
    # print(g2.es["weight"])
    # print(g2.vs["comm"])
    # print(g2.vs[_degree])

    # ic.disable()
    for i in range(10):
        # graph = ig.Graph.Famous("Cubical")
        graph = ig.Graph.Famous("zachary")
        print(leiden(graph, "comm"))
    # print(graph.vs["comm"])

    # print("###########################################")

    # for i in range(10):
    #     graph = ig.Graph.Famous("zachary")
    #     print(leidenClass(graph, "comm"))
    # # print(graph.vs["comm"])
