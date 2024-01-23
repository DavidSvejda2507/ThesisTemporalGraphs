import numpy as np
import igraph as ig
import random
from icecream import ic
from math import exp
from queue import SimpleQueue

import GraphGenerators as GrGen
import GraphAnalysis as GrAn
import matplotlib.pyplot as plt

# fmt: off
def leiden(graphs, attr, iterations, consistency_weight, initialisation = None, sanitise = True):
    alg = LeidenClass(graphs, consistency_weight)
    # ic("start")
    alg.initialiseGraph(initialisation)
    for _ in range(iterations):
        # ic("loop outer")
        alg.localMove(alg._comm)\
            .cleanCommunities(alg._comm)\
            .initialisePartition(alg._refine)\
            .refine(alg._comm, alg._refine)\
            .cleanCommunities(alg._refine)
        while not alg.converged:
            # ic("loop inner")
            alg.aggregate()\
                .localMove(alg._comm)\
                .cleanCommunities(alg._comm)\
                .refine(alg._comm, alg._refine)\
                .cleanCommunities(alg._refine)
        alg.deAggregate()
    # fmt: on
    for graph in graphs:
        graph.vs[attr] = graph.vs[alg._comm]
        for name in [alg._comm, alg._refine, alg._refineIndex, alg._queued, alg._multiplicity, alg._degree, alg._selfEdges]:
            if attr!=name:
                del graph.vs[name]
        del graph[alg._m]
        del graph[alg._n]

    if sanitise:
        alg.renumber(attr)
    del alg

    return sum(quality(graphs, attr, consistency_weight))


class LeidenClass:
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

    def __init__(self, graphs, consistency_weight=0.5, gamma=1, theta=0.01):
        self.graph_stack = [graphs]
        self.gamma = gamma
        self.theta = theta
        self.consistency_weight = consistency_weight

    def initialiseGraph(self, initialisation):
        for graph in self.graph_stack[0]:
            m2 = 0
            if graph.is_weighted():
                for vertex in graph.vs:
                    degree = 0
                    for neighbor in vertex.neighbors():
                        degree += graph[vertex.index, neighbor]
                    vertex[self._degree] = degree
                    m2 += degree
            else:
                graph.es["weight"] = 1
                graph.vs[self._degree] = graph.degree(range(graph.vcount()))
                m2 = sum(graph.vs[self._degree])
            graph.vs[self._selfEdges] = 0
            graph.vs[self._multiplicity] = 1
            graph[self._m] = m2 / 2
            graph[self._n] = graph.vcount()
            
        if initialisation is None:
            self.initialisePartition(self._comm)
        else:
            self.initialisePartitionFromAttribute(self._comm, initialisation)
        return self

    def initialisePartition(self, attr):
        communities = {}
        i = 0
        for graph in self.graph_stack[-1]:
            for vertex in graph.vs:
                vertex[attr] = i
                communities[i] = (vertex[self._multiplicity],
                                  0, vertex[self._degree])
                i += 1
        communities[-1] = (0, 0, 0)
        self.communities[attr] = communities
        return self
    
    def initialisePartitionFromAttribute(self, comm, init):
        communities = {}
        for i, graph in enumerate(self.graph_stack[0]):
            for v in graph.vs:
                v[comm] = str(i) + "_" + str(v[init])
                old = communities.get(v[comm], (0,0,0))
                neighbors = graph.vs[graph.neighbors(v.index)].select(**{self._comm + "_eq": v[comm]})
                weight = sum(graph[v, neighbor] for neighbor in neighbors)
                communities[v[comm]] = (old[0]+1, old[1]+weight, old[2] + v[self._degree])
        communities[-1] = (0,0,0)
        self.communities[comm] = communities
                
        

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

    def localMove(self, attr):
        ic("localMove")
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
            indices[length: length + l] = list(zip([index] * l, range(l)))
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

            community_edges = {-1: 0}
            for vertex in neighbors:
                comm = graph.vs[vertex][self._comm]
                community_edges[comm] = (
                    community_edges.get(comm, 0) + graph[vertex_id, vertex]
                )

            max_dq = [0]
            max_comm = current_comm
            cost_of_leaving = self.calculateDQMinus(
                attr,
                current_comm,
                graph_id,
                vertex_id,
                community_edges.get(current_comm, 0),
                degree,
            )
            for comm in community_edges:
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
                    if sum(dq) > sum(max_dq):
                        max_dq = dq
                        max_comm = comm

            if max_comm != current_comm:
                if max_comm == -1:
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
                        not graph.vs[vertex][self._queued]
                        and graph.vs[vertex][attr] != max_comm
                    ):
                        graph.vs[vertex][self._queued] = True
                        queue.put((graph_id, vertex))

            graph.vs[vertex_id][self._queued] = False
        return self

    def refine(self, attr, refine_attr):
        ic("refine")
        self.converged = True
        graphs = self.graph_stack[-1]
        communities = self.communities[attr]
        for graph_id, graph in enumerate(graphs):
            graph.vs[self._queued] = True
            for comm in communities:
                kwarg = {self._comm + "_eq": comm}
                indices = [v.index for v in graph.vs.select(**kwarg)]
                random.shuffle(indices)

                for vertex_id in indices:
                    if not graph.vs[vertex_id][self._queued]:
                        continue
                    neighbors = graph.vs[graph.neighbors(
                        vertex_id)].select(**kwarg)
                    degree = graph.vs[vertex_id][self._degree]
                    current_comm = graph.vs[vertex_id][self._refine]
                    self_edges = graph.vs[vertex_id][self._selfEdges]

                    community_edges = {}
                    neighbor = {}
                    for vertex in neighbors:
                        refine_comm = vertex[self._refine]
                        community_edges[refine_comm] = (
                            community_edges.get(refine_comm, 0)
                            + graph[vertex_id, vertex.index]
                        )
                        neighbor[refine_comm] = vertex

                    candidates = []
                    weights = []
                    cost_of_leaving = self.calculateDQMinus(
                        refine_attr,
                        current_comm,
                        graph_id,
                        vertex_id,
                        0,
                        degree,
                    )
                    for refine_comm in community_edges:
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
                        if sum(dq) > 0:
                            candidates.append((refine_comm, dq))
                            weights.append(exp(sum(dq) / self.theta))

                    if len(candidates) > 0:
                        target, dq = random.choices(candidates, weights)[0]

                        graph.vs[vertex_id][refine_attr] = target
                        self.update_communities(
                            refine_attr,
                            current_comm,
                            target,
                            community_edges,
                            graph.vs[vertex_id][self._multiplicity],
                            self_edges,
                            degree,
                        )

                        neighbor[target][self._queued] = False
                        graph.vs[vertex_id][self._queued] = False
                        self.converged = False
        return self

    def update_communities(self, attr, current, future, community_edges, multiplicity, self_edges, degree):
        communities = self.communities[attr]
        old_vertexcount, old_edgecount, old_degreesum = communities[current]
        communities[current] = (
            old_vertexcount - multiplicity,
            old_edgecount - (community_edges.get(current, 0) + self_edges),
            old_degreesum - degree,
        )
        old_vertexcount, old_edgecount, old_degreesum = communities.get(
            future, (0, 0, 0)
        )
        communities[future] = (
            old_vertexcount + multiplicity,
            old_edgecount + (community_edges.get(future, 0) + self_edges),
            old_degreesum + degree,
        )

    def calculateDQPlus(self, attr, comm, graph_id, vertex_id, edges, degree):
        vertexcount, edgecount, degreesum = self.communities[attr][comm]
        graph = self.graph_stack[-1][graph_id]
        dq = (
            edges / graph[self._m]
            - (2 * degreesum * degree) / (2 * graph[self._m]) ** 2
        )

        dConsistency = 0
        comm_members = [v.index for v in graph.vs if v[attr] == comm]
        vertex = [vertex_id]
        for graphs in self.graph_stack[:0:-1]:
            graph = graphs[graph_id]
            comm_members = sum([graph.vs[v][self._subVertices]
                                for v in comm_members], start=[])
            vertex = sum([graph.vs[v][self._subVertices]
                         for v in vertex], start=[])

        for _graph_id in self.graph_neigbors(graph_id):
            adj_comm = {}
            vertex_comm = {}
            _comm_members = comm_members.copy()
            _vertex = vertex.copy()
            for graphs in self.graph_stack[:-1]:
                graph = graphs[_graph_id]
                _comm_members = [graph.vs[v][self._refineIndex]
                                 for v in _comm_members]
                _vertex = [graph.vs[v][self._refineIndex] for v in _vertex]

            graph = self.graph_stack[-1][_graph_id]
            for v in _vertex:
                val = graph.vs[v][attr]
                vertex_comm[val] = vertex_comm.get(val, 0) + 1
            for v in _comm_members:
                val = graph.vs[v][attr]
                adj_comm[val] = adj_comm.get(val, 0) + 1
                
            for key in vertex_comm:
                dConsistency += vertex_comm[key] * adj_comm.get(key, 0) * 2
            dConsistency -= len(vertex) * len(comm_members)
            
        n = self.graph_stack[0][graph_id][self._n]
        dc = 2*dConsistency / (n * (n - 1))

        return (dq, self.consistency_weight * dc)

    def calculateDQMinus(self, attr, comm, graph_id, vertex_id, edges, degree):
        vertexcount, edgecount, degreesum = self.communities[attr][comm]
        graph = self.graph_stack[-1][graph_id]
        dq = (
            -edges / graph[self._m]
            + (2 * (degreesum - degree) * degree) / (2 * graph[self._m]) ** 2
        )

        dConsistency = 0
        comm_members = [v.index for v in graph.vs if v[attr]
                        == comm and v.index != vertex_id]
        vertex = [vertex_id]
        for graphs in self.graph_stack[:0:-1]:
            graph = graphs[graph_id]
            comm_members = sum([graph.vs[v][self._subVertices]
                                for v in comm_members], start=[])
            vertex = sum([graph.vs[v][self._subVertices]
                         for v in vertex], start=[])

        for _graph_id in self.graph_neigbors(graph_id):
            adj_comm = {}
            vertex_comm = {}
            _comm_members = comm_members.copy()
            _vertex = vertex.copy()
            for graphs in self.graph_stack[:-1]:
                graph = graphs[_graph_id]
                _comm_members = [graph.vs[v][self._refineIndex]
                                 for v in _comm_members]
                _vertex = [graph.vs[v][self._refineIndex] for v in _vertex]

            graph = self.graph_stack[-1][_graph_id]
            for v in _vertex:
                val = graph.vs[v][attr]
                vertex_comm[val] = vertex_comm.get(val, 0) + 1
            for v in _comm_members:
                val = graph.vs[v][attr]
                adj_comm[val] = adj_comm.get(val, 0) + 1

            for key in vertex_comm:
                dConsistency += vertex_comm[key] * adj_comm.get(key, 0) * 2
            dConsistency -= len(vertex) * len(comm_members)

        n = self.graph_stack[0][graph_id][self._n]
        dc = -2*dConsistency / (n * (n - 1))

        return (dq, self.consistency_weight * dc)

    def graph_neigbors(self, graph_id):
        output = []
        if graph_id > 0:
            output.append(graph_id - 1)
        if graph_id < len(self.graph_stack[0]) - 1:
            output.append(graph_id + 1)
        return output

    def aggregate(self):
        new_stack = []
        communities = self.communities[self._refine]
        for graph in self.graph_stack[-1]:
            partition = ig.VertexClustering.FromAttribute(graph, self._refine)
            aggregate_graph = partition.cluster_graph(
                {None: "first", self._multiplicity: "sum",
                    self._degree: "sum"}, "sum"
            )
            aggregate_graph.vs[self._subVertices] = list(partition)
            del partition

            _dict = {v[self._refine]: v.index for v in aggregate_graph.vs}
            graph.vs[self._refineIndex] = [_dict[ref]
                                           for ref in graph.vs[self._refine]]
            aggregate_graph.vs[self._selfEdges] = [
                communities[v[self._refine]][1] for v in aggregate_graph.vs
            ]
            new_stack.append(aggregate_graph)
        self.graph_stack.append(new_stack)
        return self

    def deAggregate(self):
        self.deAggregate_Unwind(self._comm)
        self.graph_stack = [self.graph_stack[0]]
        return self

    def deAggregate_Unwind(self, attr):
        for graphs, aggregate_graphs in zip(
            self.graph_stack[-2::-1], self.graph_stack[:0:-1]
        ):
            for graph, aggregate_graph in zip(graphs, aggregate_graphs):
                graph.vs[attr] = [
                    aggregate_graph.vs[index][attr]
                    for index in graph.vs[self._refineIndex]
                ]

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

    def stack_quality(self, attr, consistency_weight):
        self.deAggregate_Unwind(attr)
        return quality(self.graph_stack[0], attr, consistency_weight)


def quality(graphs, attr, consistency_weight):
    mod_quality = 0
    for graph in graphs:
        mod_quality += graph.modularity(graph.vs[attr])

    cons_quality = 0
    comms1 = renumber(graphs[0].vs[attr])
    for graph in graphs[1:]:
        comms2 = renumber(graph.vs[attr])
        cons_quality += GrAn.Consistency(comms1, comms2)
        comms1 = comms2
    return (mod_quality, consistency_weight * cons_quality)


def renumber(lst):
    dct = {}
    i = 0
    out = [None] * len(lst)
    for j, val in enumerate(lst):
        if dct.get(val, -1) == -1:
            dct[val] = i
            i += 1
        out[j] = dct[val]
    return out


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
        random.seed(i)
        graphs = [GrGen.GirvanNewmanBenchmark(
            7, 4 * i, density=0.8) for i in range(5)]
        random.seed(i)
        print(leiden(graphs, "comm", 2, 0.8))
        # ic(graph.vs["comm"])
        cluster = ig.VertexClustering.FromAttribute(graphs[0], "comm")
        ig.plot(cluster, "test.pdf")
    # print(graph.vs["comm"])

    # print("###########################################")

    # for i in range(10):
    #     graph = ig.Graph.Famous("zachary")
    #     print(leidenClass(graph, "comm"))
    # # print(graph.vs["comm"])
