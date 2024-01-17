import numpy as np
import igraph as ig
import random
from icecream import ic
from math import exp
from queue import SimpleQueue
from collections import namedtuple

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
        alg.localMove()\
            .cleanCommunities(alg._comm)\
            .initialisePartition(alg._refine)\
            .refine()\
            .cleanCommunities(alg._refine)
        while not alg.converged:
            # ic("loop inner")
            alg.aggregate()\
                .localMove()\
                .cleanCommunities(alg._comm)\
                .refine()\
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

    def localMove(self):
        ic("localMove")
        graphs = self.graph_stack[-1]
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
            self.localMoveVertex(graph_id, vertex_id, queue)
        return self
    
    def matchToNextGraph(self, graph_id_old, graph_id_new, vertices):
        graph = self.graph_stack[-1][graph_id_old]
        size = 0
        for v in vertices: size+=graph.vs[v][self._multiplicity]
        
        # find the list of vertices in the base graph represented by the current vertex(es)
        for graphs in self.graph_stack[:0:-1]:
            graph = graphs[graph_id_old]
            vertices = sum([graph.vs[v][self._subVertices]
                                for v in vertices], start=[])
        # Trace where the vertices end up in the next graph
        _vertices = vertices.copy()
        for graphs in self.graph_stack[:-1]:
            graph = graphs[graph_id_new]
            _vertices = [graph.vs[v][self._refineIndex]
                                for v in _vertices]
        # Count how many vertices in each aggregated vertex
        vertices_new = {}
        graph = self.graph_stack[-1][graph_id_new]
        for v in _vertices:
            vertices_new[v] = vertices_new.get(v, 0) + 1
        
        # Measure overlap
        candidates = []
        for v in vertices_new:
            candidates.append((vertices_new[v]/graph.vs[v][self._multiplicity], vertices_new[v], v))
        
        # candidates.sort(key = lambda x:x[0], reverse=True)
        # Select which vertices to add to the output
        output = []
        for overlap, _, index in candidates:
            if overlap**2 > random.uniform(0,1):
                output.append(index)
            
    
    def localMoveVertex(self, graph_id, vertex_id, queue):
        communities = self.communities[self._comm]
        graphs = self.graph_stack[-1]
        graph = graphs[graph_id]
        degree = graph.vs[vertex_id][self._degree]
        current_comm = graph.vs[vertex_id][self._comm]
        self_edges = graph.vs[vertex_id][self._selfEdges]
        neighbors = graph.neighbors(vertex_id)
        finalOption = namedtuple("finalOption", ["source_target", "target_comms", "dQ", "range_plus", "range_minus"])
        option = namedtuple("option", ["target", "graph", "target_comms", "dQ", "range_plus", "range_minus"])
        finalOptions = []
        options = []
        vertex_groups = [(graph_id, [vertex_id])]
        
        
        # Find the options and qualities for 1 slice moves
        # Initialise as final options
        community_edges = {-1: 0}
        for vertex in neighbors:
            comm = graph.vs[vertex][self._comm]
            community_edges[comm] = (
                community_edges.get(comm, 0) + graph[vertex_id, vertex]
            )
            
        cost_of_leaving = self.calculateDQMinus(
            self._comm,
            current_comm,
            graph_id,
            vertex_id,
            community_edges.get(current_comm, 0),
            degree,
        )
        finalOptions.append(finalOption(-1, {graph_id:-1}, sum(cost_of_leaving), graph_id, graph_id))
        options.append(option(-1, graph_id, {graph_id:-1}, sum(cost_of_leaving), graph_id, graph_id))
        
        for comm in community_edges:
            if comm != current_comm:
                dq = (
                    self.calculateDQPlus(
                        self._comm,
                        comm,
                        graph_id,
                        vertex_id,
                        community_edges[comm],
                        degree,
                    )
                    + cost_of_leaving
                )
                finalOptions.append(finalOption(comm, {graph_id:comm}, sum(dq), graph_id, graph_id))
                options.append(option(comm, graph_id, {graph_id:comm}, sum(dq), graph_id, graph_id))
        
        # Find the options and modularities for first move
        vertices = self.matchToNextGraph(graph_id, graph_id+1, [vertex_id])
        vertex_groups.append((graph_id+1, vertices))
        graph = self.graph_stack[-1][graph_id+1]
        
        current_comm_connections = {} # Only connections between selected vertices and other vertices of their current comm (by comm)
        comm_connections = {} # All connections between selected vertices and other vertices (by comm)
        current_comm_degrees = {} # Total degree of all selected vertices by comm
        internal_edges = 0 # Total weight of edges between vertices in the selection, which are in different communities
        for vertex in vertices:
            neighbors = graph.neighbors(vertex)
            current_comm = graph.vs[vertex][self._comm]
            current_degree = graph.vs[vertex][self._degree]
            current_comm_degrees[current_comm] = current_comm_degrees.get(current_comm, 0) + current_degree
            for neighbor in neighbors:
                neighbor_comm = graph.vs[neighbor][self._comm]
                if neighbor not in vertices:
                    if neighbor_comm == current_comm:
                        degree, edges = current_comm_connections.get(neighbor_comm, (0,0))
                        current_comm_connections[neighbor_comm] = (degree + current_degree, edges + graph[vertex, neighbor])
                    degree, edges = comm_connections.get(neighbor_comm, (0,0))
                    comm_connections[neighbor_comm] = (degree + current_degree, edges + graph[vertex, neighbor])
                else:
                    if neighbor_comm != current_comm:
                        internal_edges += graph[vertex, neighbor]
        internal_edges /= 2 # Correct for double counting of internal edges
                        
        # Calculate the dq of leaving the current communities
        cost_of_leaving_q = 0
        temp_communities = {}
        m = graph[self._m]
        for comm in current_comm_connections:
            degree, edges = current_comm_connections[comm]
            _, _, degree_sum = self.communities[self._comm][comm]
            cost_of_leaving_q += -edges / m + (2 * (degree_sum - degree) * degree) / (2 * m) ** 2
            temp_communities[comm] = degree_sum-degree
        # Add the dq of the different vertices comming together
        cost_of_leaving_q += internal_edges / m
        total_degree = 0
        degrees_squared = 0
        for comm in current_comm_degrees:
            degree = current_comm_degrees[comm]
            degrees_squared += degree * degree
            total_degree += degree
        cost_of_leaving_q += (degrees_squared-total_degree**2)/((2*m)**2)
        
        # Find the consistencies for each pair of 0- and 1-moves
        # Update current options to new layer
        # Update final options with potential improvements
        # Do not discard bad moves
        
        # Find the options and modularities for second move 
        
        # Find the consistencies for each pair of 1- and 2-moves
        # Update current options to new layer
        # Update final options with potential improvements
        # Do not discard bad moves
        
        
            
    
    def localMoveVertex(self, graph_id, vertex_id, queue):
        communities = self.communities[self._comm]
        graphs = self.graph_stack[-1]
        graph = graphs[graph_id]
        degree = graph.vs[vertex_id][self._degree]
        current_comm = graph.vs[vertex_id][self._comm]
        self_edges = graph.vs[vertex_id][self._selfEdges]
        neighbors = graph.neighbors(vertex_id)
        

        vertices_under_consideration = {0:[vertex_id]}
        community_edges_dict = {0:community_edges}
        source_result = namedtuple("result", ["target_comm", "dQ", "list_plus", "list_minus"])
        result = namedtuple("result", ["target_comm", "graph_offset", "previous_target", "dQ"])
        results = [source_result(current_comm, 0, None, 0, [], [])]
        
        community_edges = {-1: 0}
        for vertex in neighbors:
            comm = graph.vs[vertex][self._comm]
            community_edges[comm] = (
                community_edges.get(comm, 0) + graph[vertex_id, vertex]
            )
            
        cost_of_leaving = self.calculateDQMinus(
            self._comm,
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
                        self._comm,
                        comm,
                        graph_id,
                        vertex_id,
                        community_edges[comm],
                        degree,
                    )
                    + cost_of_leaving
                )
                if sum(dq) > 0 or dq[0] + dq[2] > 0:
                    max_dq = dq
                    max_comm = comm


        if max_comm != current_comm:
            if max_comm == -1:
                i = 0
                communities = self.communities[self._comm]
                while True:
                    if communities.get(i, (0, 0, 0))[0] == 0:
                        break
                    i += 1
                max_comm = i
            graph.vs[vertex_id][self._comm] = max_comm
            self.update_communities(
                self._comm,
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
                    and graph.vs[vertex][self._comm] != max_comm
                ):
                    graph.vs[vertex][self._queued] = True
                    queue.put((graph_id, vertex))

        graph.vs[vertex_id][self._queued] = False
        

    def refine(self):
        ic("refine")
        self.converged = True
        graphs = self.graph_stack[-1]
        communities = self.communities[self._comm]
        for graph_id, graph in enumerate(graphs):
            graph.vs[self._queued] = True
            for comm in communities:
                kwarg = {self._comm + "_eq": comm}
                indices = [v.index for v in graph.vs.select(**kwarg)]
                random.shuffle(indices)

                for vertex_id in indices:
                    if not graph.vs[vertex_id][self._queued]:
                        continue
                    self.refineVertex(graphs, vertex_id, graph_id, kwarg)
        return self
    
    def refineVertex(self, graph, vertex_id, graph_id, kwarg):
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
            self._refine,
            current_comm,
            graph_id,
            vertex_id,
            0,
            degree,
        )
        for refine_comm in community_edges:
            dq = (
                self.calculateDQPlus(
                    self._refine,
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

            neighbor[target][self._queued] = False
            graph.vs[vertex_id][self._queued] = False
            self.converged = False

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
        
    def deAggregateVertex(self, graph_id, *vertices):
        for graphs in self.graph_stack[:0:-1]:
            subvertices = graphs[graph_id].vs[self._subVertices]
            vertices = tuple(sum([subvertices[v] for v in item], start=[]) 
                             for item in vertices)
        return vertices
        
    def reAggregateVertex(self, graph_id, *vertices):
        for graphs in self.graph_stack[:-1]:
            refineIndices = graphs[graph_id].vs[self._refineIndex]
            vertices = tuple([refineIndices[v] for v in item] 
                             for item in vertices)
        return vertices
            
    def countCommunities(self, graph, attr, *vertices):
        output = []
        lookup = graph.vs[attr]
        for _list in vertices:
            _dict = {}
            for v in _list:
                val = lookup[v]
                _dict[val] = _dict.get(val, 0) + 1
            output.append(_dict)
        return output

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
        comm_members, vertex = self.deAggregateVertex(graph_id, comm_members, vertex)

        for _graph_id in self.graph_neigbors(graph_id):
            _comm_members = comm_members.copy()
            _vertex = vertex.copy()
            _comm_members, _vertex = self.reAggregateVertex(_graph_id, _comm_members, _vertex)

            graph = self.graph_stack[-1][_graph_id]
            adj_comm, vertex_comm = self.countCommunities(graph, attr, _comm_members, _vertex)
                
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
        comm_members, vertex = self.deAggregateVertex(graph_id, comm_members, vertex)

        for _graph_id in self.graph_neigbors(graph_id):
            _comm_members = comm_members.copy()
            _vertex = vertex.copy()
            _comm_members, _vertex = self.reAggregateVertex(_graph_id, _comm_members, _vertex)

            graph = self.graph_stack[-1][_graph_id]
            adj_comm, vertex_comm = self.countCommunities(graph, attr, _comm_members, _vertex)

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
            7, 4 * i, density=0.5) for i in range(3)]
        random.seed(i)
        print(leiden(graphs, "comm", 1, 0.564843))
        # ic(graph.vs["comm"])
        cluster = ig.VertexClustering.FromAttribute(graphs[0], "comm")
        ig.plot(cluster, "test.pdf")
    # print(graph.vs["comm"])

    # print("###########################################")

    # for i in range(10):
    #     graph = ig.Graph.Famous("zachary")
    #     print(leidenClass(graph, "comm"))
    # # print(graph.vs["comm"])
