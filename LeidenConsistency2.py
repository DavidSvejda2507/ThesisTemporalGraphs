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
        for name in [alg._comm, alg._refine, alg._refineIndex, alg._queued, alg._inQueue, alg._multiplicity, alg._degree, alg._selfEdges]:
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
    _inQueue = "leiden_inQueue"
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
        self.counter = 0

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
            graph.vs[self._inQueue] = True

        while not queue.empty():
            graph_id, vertex_id = queue.get()
            self.localMoveVertex(graph_id, vertex_id, queue)
        return self
    
    def matchToNextGraph(self, graph_id_old, graph_id_new, vertex):
        # find the list of vertices in the base graph represented by the current vertex(es)
        vertices, = self.deAggregateVertex(graph_id_old, [vertex])
        # Trace where the vertices end up in the next graph
        vertices, = self.reAggregateVertex(graph_id_new, vertices)
        # Count how many vertices in each aggregated vertex
        vertices_new = {}
        graph = self.graph_stack[-1][graph_id_new]
        for v in vertices:
            vertices_new[v] = vertices_new.get(v, 0) + 1
        
        # Measure overlap
        candidates = []
        weights = []
        for v in vertices_new:
            candidates.append(v)
            weights.append(vertices_new[v]**2/graph.vs[v][self._multiplicity])
        
        # candidates.sort(key = lambda x:x[0], reverse=True)
        # Select which vertices to add to the output
        return random.choices(candidates, weights)[0]
      
    def findMoves(self, graph, vertex_id, current_graph_id, community_edges_dict):
        set_list = namedtuple("set_list", ["vertices", "old_comm", "new_comms", "set_names"],)
    
        degree = graph.vs[vertex_id][self._degree]
        current_comm = graph.vs[vertex_id][self._comm]
        neighbors = graph.neighbors(vertex_id)
        
        community_edges = {-1: 0}
        for vertex in neighbors:
            comm = graph.vs[vertex][self._comm]
            community_edges[comm] = (
                community_edges.get(comm, 0) + graph[vertex_id, vertex]
            )
        community_edges_dict[current_graph_id] = community_edges
            
        cost_of_leaving, current_comm_members, vertex_members = self.calculateDQMinus(
            self._comm,
            current_comm,
            current_graph_id,
            vertex_id,
            community_edges.get(current_comm, 0),
            degree,
        )
        local_options = {-1:cost_of_leaving}
        
        neighbor_comms = {}
        set_names = [-1]
        next_comms = set_list(frozenset(vertex_members), frozenset(current_comm_members), neighbor_comms, set_names)
        
        for comm in community_edges:
            if comm != current_comm and comm != -1:
                dq, neighbor_comm = self.calculateDQPlus(
                        self._comm,
                        comm,
                        current_graph_id,
                        vertex_id,
                        community_edges[comm],
                        degree,
                    )
                local_options[comm] = dq + cost_of_leaving
                set_names.append(comm)
                neighbor_comms[comm] = frozenset(neighbor_comm)

        return local_options, next_comms
 
    def cross_intersect(self, vertex1, comm1, vertex2, comm2):
        vv = len(vertex1.intersection(vertex2))
        vc = len(vertex1.intersection(comm2))
        cv = len(comm1.intersection(vertex2))
        cc = len(comm1.intersection(comm2))
        return vv*cc + vc*cv
           
    def calculate_interactions(self, set_list1, set_list2, normalisation_factor):
        output = {}
        for key1 in set_list1.set_names:
            output[key1] = {}
        
        output[-1][-1] = self.cross_intersect(set_list1.vertices, set_list1.old_comm, set_list2.vertices, set_list2.old_comm)
        for key2 in set_list2.set_names[1:]:
            intersection = self.cross_intersect(set_list1.vertices, set_list1.old_comm, set_list2.vertices, set_list2.new_comms[key2])
            output[-1][key2] = output[-1][-1] - intersection
            
        for key1 in set_list1.set_names[1:]:
            intersection = self.cross_intersect(set_list1.vertices, set_list1.new_comms[key1], set_list2.vertices, set_list2.old_comm)
            output[key1][-1] = output[-1][-1] - intersection
            for key2 in set_list2.set_names[1:]:
                intersection = self.cross_intersect(set_list1.vertices, set_list1.new_comms[key1], set_list2.vertices, set_list2.new_comms[key2])
                output[key1][key2] = -output[-1][-1] + output[-1][key2] + output[key1][-1] + intersection
            
        normalisation_factor *= 2
        normalisation_factor *= self.consistency_weight
        for key1 in set_list1.set_names:
            for key2 in set_list2.set_names:
                output[key1][key2] *= normalisation_factor
        
        return output
           
    def updateOptions(self, final_options, intermediate_options, local_options, interactions, graph_id):
        final_option = namedtuple("final_option", ["source_target", "target_comms", "dQ"])
        option = namedtuple("option", ["target", "graph", "source_target", "target_comms", "dQ"])
        next_options = []
        for index, final_opt in enumerate(final_options):
            for local_opt in local_options:
                max_dQ = -1e10
                best_intermediate = None
                intermediate_options = (opt for opt in intermediate_options if opt.source_target == final_opt.source_target)
                for inter_opt in intermediate_options:
                    dQ = inter_opt.dQ + sum(local_options[local_opt]) + interactions[inter_opt.target][local_opt]
                    dq = sum(local_options[local_opt][0::2])
                    dc = sum(local_options[local_opt][1::2]) + interactions[inter_opt.target][local_opt]
                    if (dq>0 or dc>0) and dQ > max_dQ:
                        max_dQ = dQ
                        best_intermediate = inter_opt
                if best_intermediate is not None:
                    targets = best_intermediate.target_comms.copy()
                    targets[graph_id] = local_opt
                    new_opt = option(local_opt, graph_id, final_opt.source_target, targets, max_dQ)
                    next_options.append(new_opt)
                    if max_dQ > final_opt.dQ:
                        final_opt = final_option(final_opt.source_target, targets, max_dQ)
                        final_options[index] = final_opt
        return next_options

    def localMoveVertex(self, graph_id, vertex_id, queue):
        graphs = self.graph_stack[-1]
        graph = graphs[graph_id]
        
        graph.vs[vertex_id][self._inQueue] = False
        if not graph.vs[vertex_id][self._queued]:
            return
        
        consistency_normalisation_factor = 2/(graph[self._n]*(graph[self._n]-1))
        selected_vertices = {graph_id: vertex_id}
        community_edges_dict = {}
        
        # Find the options and qualities for 1 slice moves
        local_options, base_comms = self.findMoves(graph, vertex_id, graph_id, community_edges_dict)
        
        # Initialise as final options
        final_option = namedtuple("final_option", ["source_target", "target_comms", "dQ"])
        final_options = []
        option = namedtuple("option", ["target", "graph", "source_target", "target_comms", "dQ"])
        options = []
        
        for comm in local_options:
            final_options.append(final_option(comm, {graph_id:comm}, sum(local_options[comm])))
            options.append(option(comm, graph_id, comm, {graph_id:comm}, sum(local_options[comm])))
            
        
        for direction, limit in [(-1, -1), (1, len(self.graph_stack[0]))]:
            previous_comms = base_comms
            previous_graph_id = graph_id
            vertex_id = selected_vertices[graph_id]
            for current_graph_id in range(graph_id+direction, limit, direction):
                
                # Find the options and modularities for moves
                vertex_id = self.matchToNextGraph(previous_graph_id, current_graph_id, vertex_id)
                selected_vertices[current_graph_id] = vertex_id
                graph = self.graph_stack[-1][current_graph_id]
                
                local_options, next_comms = self.findMoves(graph, vertex_id, current_graph_id, community_edges_dict)
                
                # Find the consistencies for each pair of moves
                interactions = self.calculate_interactions(previous_comms, next_comms, consistency_normalisation_factor)
                
                # Update current options to new layer
                # Update final options with potential improvements
                options = self.updateOptions(final_options, options, local_options, interactions, current_graph_id)
                
                if len(options) < 1: break
                previous_comms = next_comms
                previous_graph_id = current_graph_id
                
        final_option = max(final_options, key = lambda x:x.dQ)
        if final_option.dQ <= 0:
            graphs[graph_id].vs[selected_vertices[graph_id]][self._queued] = False
            return
            
        communities = self.communities[self._comm]
        # quality_old = self.stack_quality(self._comm, self.consistency_weight)
        for graph_id in final_option.target_comms:
            graph = graphs[graph_id]
            vertex_id = selected_vertices[graph_id]
            target_comm = final_option.target_comms[graph_id]
            
            if target_comm == -1:
                i = 0
                while True:
                    if communities.get(i, (0, 0, 0))[0] == 0:
                        break
                    i += 1
                target_comm = i
            
            self.update_communities(
                self._comm,
                graph.vs[vertex_id][self._comm],
                target_comm,
                community_edges_dict[graph_id],
                graph.vs[vertex_id][self._multiplicity],
                graph.vs[vertex_id][self._selfEdges],
                graph.vs[vertex_id][self._degree],
            )
            graph.vs[vertex_id][self._comm] = target_comm

            for vertex in graph.neighbors(vertex_id):
                if (
                    not graph.vs[vertex][self._queued]
                    and graph.vs[vertex][self._comm] != target_comm
                ):
                    graph.vs[vertex][self._queued] = True
                    if not graph.vs[vertex][self._inQueue]:
                        queue.put((graph_id, vertex))

            graph.vs[vertex_id][self._queued] = False
        # quality_new = self.stack_quality(self._comm, self.consistency_weight)
        # dQuality = sum(quality_new) - sum(quality_old)
        # if dQuality < 0:
        #     ic(dQuality)
       
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
                    self.refineVertex(graph, vertex_id, graph_id, kwarg)
        return self
    
    def refineVertex(self, graph, vertex_id, graph_id, kwarg):
        neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
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
        cost_of_leaving, _, _ = self.calculateDQMinus(
            self._refine,
            current_comm,
            graph_id,
            vertex_id,
            0,
            degree,
        )
        for refine_comm in community_edges:
            dq, _ = self.calculateDQPlus(
                    self._refine,
                    refine_comm,
                    graph_id,
                    vertex_id,
                    community_edges[refine_comm],
                    degree,
                )
            dq += cost_of_leaving
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

    def calculateDQPlus(self, attr, comm, graph_id, vertex_id, edges, degree, deaggregated_vertex = None):
        vertexcount, edgecount, degreesum = self.communities[attr][comm]
        graph = self.graph_stack[-1][graph_id]
        dq = (
            edges / graph[self._m]
            - (2 * degreesum * degree) / (2 * graph[self._m]) ** 2
        )

        dConsistency = 0
        comm_members = [v.index for v in graph.vs if v[attr] == comm]
        if deaggregated_vertex is None:
            vertex = [vertex_id]
            comm_members, vertex = self.deAggregateVertex(graph_id, comm_members, vertex)
        else:
            comm_members = self.deAggregateVertex(graph_id, comm_members)
            vertex = deaggregated_vertex

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

        return (dq, self.consistency_weight * dc), comm_members

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

        return (dq, self.consistency_weight * dc), comm_members, vertex

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
            7, 4 * i, density=0.8) for i in range(10)]
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
