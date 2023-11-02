import numpy as np
import igraph as ig
import random
from icecream import ic
from math import exp
from queue import SimpleQueue

import GraphGenerators as GrGen
import matplotlib.pyplot as plt


class LeidenAlg:
    _comm = "leiden_commLeiden"
    _refine = "leiden_commLeidenRefinement"
    _refineIndex = "leiden_refinementIndex"
    _queued = "leiden_queued"
    _multiplicity = "leiden_multiplicity"
    _degree = "leiden_degree"
    _selfEdges = "leiden_selfEdges"
    _subVertices = "leiden_subVertices"
    _m = "leiden_m"
    _n = "leiden_n"
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
                communities[i] = (vertex[self._multiplicity], 0, graph.degree(vertex))
                i += 1
        communities[-1] = (0, 0, 0)
        self.communities[attr] = communities
        return self

    def cleanCommunities(self, attr):
        output = {}
        for key in self.communities[attr]:
            val = self.communities[attr][key]
            if val[0] != 0:
                output[key] = val
            else:
                if val[1] != 0 or val[2] != 0:
                    raise ValueError(
                        f"Community with {val[1]} internal edges and {val[2]} internal degree without internal vertices found"
                    )
        output[-1] = (0, 0, 0)
        self.communities[attr] = output
        return self

    def calculateDQPlus(self, attr, comm, graph_id, vertex_id, edges, degree):
        vertexcount, edgecount, degreesum = self.communities[attr][comm]
        dq = (
            edges / self.graph_stack[-1][graph_id][self._m]
            - ((2 * degreesum + degree) * degree)
            / (2 * self.graph_stack[-1][graph_id][self._m]) ** 2
        )

        # unaggregated graph:
        dConsistency = 0
        current = [comm == _comm for _comm in self.graph_stack[-1][graph_id].vs[attr]]
        for _graph_id in self.graph_neigbors(graph_id):
            graph = self.graph_stack[-1][_graph_id]
            adj_comm = graph.vs[vertex_id][attr]
            adjacent = [adj_comm == _comm for _comm in graph.vs[attr]]
            dConsistency += sum([curr and adj for curr, adj in zip(current, adjacent)])

        n = self.graph_stack[0][graph_id][self._n]
        dc = 2 * dConsistency / (n * (n + 1))

        return dq + dc

    def calculateDQMinus(self, attr, comm, graph_id, vertex_id, edges, degree):
        vertexcount, edgecount, degreesum = self.communities[attr][comm]
        dq = (
            -edges / self.graph_stack[-1][graph_id][self._m]
            + ((2 * degreesum - degree) * degree)
            / (2 * self.graph_stack[-1][graph_id][self._m]) ** 2
        )

        # unaggregated graph:
        dConsistency = 0
        comm_members = [
            comm == _comm for _comm in self.graph_stack[-1][graph_id].vs[attr]
        ]
        comm_members[vertex_id] = False
        for graphs in self.graph_stack[-2::-1]:
            comm_members = [
                comm_members[index] for index in graphs[graph_id].vs[self._refineIndex]
            ]
        vertex = [vertex_id]
        for graph in self.graph_stack[:0:-1]:
            vertex = sum([graph[v][self._subVertices] for v in vertex], start=[])
        for _graph_id in self.graph_neigbors(graph_id):
            adj_comm = {}
            for graphs in self.graph_stack[:-1]:
                graph = graphs[_graph_id]

            graph = self.graph_stack[-1][_graph_id]
            adjacent = [adj_comm == _comm for _comm in graph.vs[attr]]
            for curr, adj in zip(comm_members, adjacent):
                dConsistency += curr * (1 - 2 * adj)

        n = self.graph_stack[0][graph_id][self._n]
        dc = -2 * dConsistency / (n * (n + 1))

        return dq + dc

    def graph_neigbors(self, graph_id):
        output = []
        if graph_id > 0:
            output.append(graph_id - 1)
        if graph_id < len(self.graph_stack[0]) - 1:
            output.append(graph_id + 1)
        return output

    def checkWeights(self):
        for graph in self.graph_stack[0]:
            m = 0
            if "weight" in graph.es.attributes():
                for vertex in graph.vs:
                    degree = 0
                    for neighbor in vertex.neighbors():
                        degree += graph[vertex.index, neighbor]
                    vertex[self._degree] = degree
                    m += degree
            else:
                graph.es["weight"] = 1
                graph.vs[self._degree] = graph.degree(range(graph.vcount()))
                m = sum(graph.vs[self._degree])
            graph.vs[self._selfEdges] = 0
            graph.vs[self._multiplicity] = 1
            graph[self._m] = m / 2
            graph[self._n] = graph.vcount()
        return self

    def aggregate(self):
        new_stack = []
        communities = self.communities[self._refine]
        for graph in self.graph_stack[-1]:
            partition = ig.VertexClustering.FromAttribute(graph, self._refine)
            aggregate_graph = partition.cluster_graph(
                {None: "first", self._multiplicity: "sum", self._degree: "sum"}, "sum"
            )
            aggregate_graph.vs[self._subVertices] = list(partition)
            del partition

            _dict = {v[self._refine]: v.index for v in aggregate_graph.vs}
            graph.vs[self._refineIndex] = [_dict[ref] for ref in graph.vs[self._refine]]
            aggregate_graph.vs[self._selfEdges] = [
                communities[v[self._refine]][1] for v in aggregate_graph.vs
            ]
            new_stack.append(aggregate_graph)
        self.graph_stack.append(new_stack)
        return self

    def deAggregate(self):
        for graphs, aggregate_graphs in zip(
            self.graph_stack[-2::-1], self.graph_stack[:0:-1]
        ):
            for graph, aggregate_graph in zip(graphs, aggregate_graphs):
                graph.vs[self._comm] = [
                    aggregate_graph.vs[index][self._comm]
                    for index in graph.vs[self._refineIndex]
                ]
        return self

    def renumber(self, attr):
        _dict = {}
        i = 0
        for graph in self.graph_stack[0]:
            for c in graph.vs[attr]:
                val = _dict.get(c, -1)
                if val == -1:
                    _dict[c] = i
                    i += 1
        for graph in self.graph_stack[0]:
            graph.vs[attr] = [_dict[c] for c in graph.vs[attr]]
        return self

    def update_communities(
        self, attr, current, future, community_edges, multiplicity, self_edges, degree
    ):
        communities = self.communities[attr]
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

    def localMove(self, attr):
        graphs = self.graph_stack[-1]
        communities = self.communities[attr]
        queue = SimpleQueue()
        length = 0
        for graph in graphs:
            length += graph.vcount()
        indices = [None] * length
        length = 0
        for index, graph in enumerate(graphs):
            l = graph.vcount()
            indices[length : length + l] = list(zip([index] * l, range(l)))
            length += l
        random.shuffle(indices)
        for i in indices:
            queue.put(i)
        for graph in graphs:
            graph.vs[self._queued] = True

        while not queue.empty():
            graph_id, vertex_id = queue.get()
            graph = graphs[graph_id]
            degree = graph.vs[vertex_id][self._degree]
            current_comm = graph.vs[vertex_id][self._comm]
            self_edges = graph.vs[vertex_id][self._selfEdges]
            neighbors = graph.neighbors(vertex_id)
            community_edges = {-1: self_edges}
            for vertex in neighbors:
                comm = graph.vs[vertex][self._comm]
                community_edges[comm] = (
                    community_edges.get(comm, self_edges) + graph[vertex_id, vertex]
                )

            max_dq = 0
            max_comm = current_comm
            cost_of_leaving = self.calculateDQMinus(
                attr,
                current_comm,
                graph_id,
                vertex_id,
                community_edges.get(current_comm, self_edges),
                degree,
            )
            for comm in community_edges.keys():
                if comm != current_comm:
                    dq = (
                        self.calculateDQPlus(
                            attr,
                            comm,
                            graph_id,
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
                if max_comm == -1:
                    ic("split")
                    i = 0
                    communities = self.communities[attr]
                    while True:
                        if communities.get(i, (0, 0, 0))[0] == 0:
                            break
                        i += 1
                    max_comm = i
                graph.vs[vertex_id][attr] = max_comm
                self.update_communities(
                    attr,
                    current_comm,
                    max_comm,
                    community_edges,
                    graph.vs[vertex_id][self._multiplicity],
                    self_edges,
                    degree,
                )

                for vertex in neighbors:
                    if (
                        graph.vs[vertex][attr] != max_comm
                        and not graph.vs[vertex][self._queued]
                    ):
                        graph.vs[vertex][self._queued] = True
                        queue.put((graph_id, vertex))

            graph.vs[vertex_id][self._queued] = False
        return self

    def refine(self, attr, refine_attr):
        self.converged = True
        graphs = self.graph_stack[-1]
        communities = self.communities[attr]
        refine_communities = self.communities[refine_attr]
        for graph_id, graph in enumerate(graphs):
            for comm in communities:
                kwarg = {self._comm + "_eq": comm}
                indices = [v.index for v in graph.vs.select(**kwarg)]
                random.shuffle(indices)

                for vertex_id in indices:
                    if refine_communities[graph.vs[vertex_id][self._refine]][0] > 1:
                        continue
                    neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
                    degree = graph.vs[vertex_id][self._degree]
                    current_comm = graph.vs[vertex_id][self._refine]
                    self_edges = graph.vs[vertex_id][self._selfEdges]
                    community_edges = {}
                    for vertex in neighbors:
                        refine_comm = vertex[self._refine]
                        community_edges[refine_comm] = (
                            community_edges.get(refine_comm, self_edges)
                            + graph[vertex_id, vertex.index]
                        )

                    candidates = []
                    weights = []
                    cost_of_leaving = self.calculateDQMinus(
                        refine_attr,
                        current_comm,
                        graph_id,
                        vertex_id,
                        community_edges.get(current_comm, self_edges),
                        degree,
                    )
                    for refine_comm in community_edges.keys():
                        dq = (
                            self.calculateDQPlus(
                                refine_attr,
                                refine_comm,
                                graph_id,
                                vertex_id,
                                community_edges[refine_comm],
                                degree,
                            )
                            + cost_of_leaving
                        )
                        if dq > 0:
                            candidates.append(refine_comm)
                            weights.append(exp(dq / self.theta))

                    if len(candidates) > 0:
                        target = random.choices(candidates, weights)[0]
                        graph.vs[vertex_id][self._refine] = target
                        self.update_communities(
                            self._refine,
                            current_comm,
                            target,
                            community_edges,
                            graph.vs[vertex_id][self._multiplicity],
                            self_edges,
                            degree,
                        )
                        self.converged = False
        return self


# fmt: off
def leidenClass(graphs, attr, iterations):
    alg = LeidenAlg(graphs)
    ic("start")
    alg.checkWeights()\
        .initialisePartition(alg._comm)
    for _ in range(iterations):
        ic("loop outer")
        alg.localMove(alg._comm)\
            .cleanCommunities(alg._comm)\
            .initialisePartition(alg._refine)\
            .refine(alg._comm, alg._refine)\
            .cleanCommunities(alg._refine)
        while not alg.converged:
            ic("loop inner")
            alg.aggregate()\
                .localMove(alg._comm)\
                .cleanCommunities(alg._comm)\
                .refine(alg._comm, alg._refine)\
                .cleanCommunities(alg._refine)\
                .deAggregate()
    # fmt: on
    for graph in graphs:
        graph.vs[attr] = graph.vs[alg._comm]
        del graph.vs[alg._comm]
        del graph.vs[alg._refine]
        del graph.vs[alg._refineIndex]
        del graph.vs[alg._selfEdges]
        del graph.vs[alg._multiplicity]
        del graph.vs[alg._degree]
        del graph.vs[alg._queued]

    alg.renumber(attr)
    del alg

    return [quality(graphs[i], attr) for i in range(len(graphs))]

def quality(graph, comm):
    return graph.modularity(graph.vs[comm])

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
        graphs = [GrGen.GirvanNewmanBenchmark(7, 4*i, density=0.5) for i in range(3)]
        random.seed(36)
        print(leidenClass(graphs, "comm", 10))
        # ic(graph.vs["comm"])
        cluster = ig.VertexClustering.FromAttribute(graphs[0], "comm")
        ig.plot(cluster, "test.pdf")
    # print(graph.vs["comm"])

    # print("###########################################")

    # for i in range(10):
    #     graph = ig.Graph.Famous("zachary")
    #     print(leidenClass(graph, "comm"))
    # # print(graph.vs["comm"])
