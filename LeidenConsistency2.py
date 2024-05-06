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
def leiden(graphs, attr, iterations, consistency_weight, initialisation = None, sanitise = True, refinement_consistency_refference = "refine"):
    """Performs the second version of the consistency leiden algorithm.

    Args:
        graphs (list[ig.graph]): List of graphs to be clustered.
        attr (str): Name of the vertex attribute where the result of the clustering will be saved.
        iterations (int): Number of iterations that the algorithms performs.
        consistency_weight (float): Weight of the consistency relative to the modularity.
        initialisation (string, optional): Name of vertex attribute that stores the initial communities. Defaults to None.
        sanitise (bool, optional): Whether to renumber the output communities to 0 ... k. Defaults to True.
        refinement_consistency_refference ("refine" or "comm", optional): Which community structure to use when calculating the consistency during the refinement step. Defaults to "refine"

    Returns:
        float: Quality of the found communities.
    """    
    alg = LeidenClass(graphs, consistency_weight, refinement_consistency_refference = refinement_consistency_refference)
    alg.initialiseGraph(initialisation)
    for _ in range(iterations):
        alg.localMove()\
            .cleanCommunities(alg._comm)\
            .initialisePartition(alg._refine)\
            .refine()\
            .cleanCommunities(alg._refine)
        while not alg.converged:
            alg.aggregate()\
                .localMove()\
                .cleanCommunities(alg._comm)\
                .refine()\
                .cleanCommunities(alg._refine)
        alg.deAggregate()
    # fmt: on
    for graph in graphs:
        graph.vs[attr] = graph.vs[alg._comm]
        for name in [alg._comm, alg._refine, alg._refineIndex, alg._wellConnected, alg._queued, alg._inQueue, alg._multiplicity, alg._degree, alg._selfEdges]:
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
    _wellConnected = "leiden_wellConnected"
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

    def __init__(self, graphs, consistency_weight=0.5, gamma=1, theta=0.01, refinement_consistency_refference = "refine"):
        """Make a LeidenClass object of use in the consistency leiden algorithm 2.

        Args:
            graphs (list[ig.Graph]): The sequence of graphs to cluster.
            consistency_weight (float, optional): Weight of the consistency relative to the modularity. Defaults to 0.5.
            gamma (float, optional): UNIMPLEMENTED Scale of modularity. Defaults to 1.
            theta (float, optional): Parameter that affects the randomness in the refinement step. Defaults to 0.01.
            refinement_consistency_refference ("refine" or "comm", optional): Which community structure to use when calculating the consistency during the refinement step. Defaults to "refine"
        """        
        self.graph_stack = [graphs]
        self.gamma = gamma
        self.theta = theta
        self.consistency_weight = consistency_weight
        self.counter = 0
        self._refineConsistency = {"comm":self._comm, "refine":self._refine}.get(refinement_consistency_refference)

    def initialiseGraph(self, initialisation=None):
        """Prepares the graphs by storing info in attributes.

        Args:
            initialisation (string, optional): Name of the attribute storing the initialisation. Defaults to None.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """        
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
        """Initialises the singleton partition.

        Args:
            attr (string): Name of the vertex attribute in which to store the community membership.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """        
        communities = {}
        i = 0
        for index, graph in enumerate(self.graph_stack[-1]):
            for vertex in graph.vs:
                vertex[attr] = i
                communities[i] = (vertex[self._multiplicity],
                                  0, vertex[self._degree], index, True)
                i += 1
        communities[-1] = (0, 0, 0, -1, True)
        self.communities[attr] = communities
        return self
    
    def initialisePartitionFromAttribute(self, comm, init):
        """Initialises a partition from an existing partition.

        Args:
            comm (string): Name of the vertex attribute in which to store the community membership.
            init (string): Name of the vertex attribute in which the existing partition is stored.
        """        
        communities = {}
        for i, graph in enumerate(self.graph_stack[0]):
            for v in graph.vs:
                v[comm] = str(i) + "_" + str(v[init])
                old = communities.get(v[comm], (0,0,0,-1))
                neighbors = graph.vs[graph.neighbors(v.index)].select(**{self._comm + "_eq": v[comm]})
                weight = sum(graph[v, neighbor] for neighbor in neighbors)
                communities[v[comm]] = (old[0]+1, old[1]+weight, old[2] + v[self._degree], i, True)
        communities[-1] = (0, 0, 0, -1, True)
        self.communities[comm] = communities
                
    def cleanCommunities(self, attr):
        """Remove empty communities from the community dictionary.

        Args:
            attr (string): Name of community dict to clean.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """        
        output = {}
        communities = self.communities[attr]
        for key in communities:
            val = communities[key]
            if val[0] != 0:
                output[key] = val
        output[-1] = (0, 0, 0, -1, True)
        self.communities[attr] = output
        return self

    def localMove(self):
        """Perform the local move step of the algorithm.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """         
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
        """Matches a vertex to a vertex in another graph, where vertices with more overlap have a larger chance of being selected.

        Args:
            graph_id_old (int): Index of the graph in which the initial vertex lives.
            graph_id_new (int): Index of the grpah in which to look for a matching vertex.
            vertex (int): Index of the initial vertex.

        Returns:
            int: Index of the found vertex in the new graph.
        """        
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
        """Finds all of the possible communities into which the vertex can be moved, and calculates the change in quality for each possible move, assuming only this vertex is moved.

        Args:
            graph (ig.Graph): The current graph.
            vertex_id (int): Index of the vertex.
            current_graph_id (int): Index of the current graph.
            community_edges_dict (dict[int:dict]): Dictionary into which the summary of connections to adjacent communities is to be put.

        Returns:
            local_options (dict[int:tuple]): Dictionary of target communities and their corresponding changes in quality.
            next_comms (set_list): Named tuple with sets containing the indices of the vertices in the current vertex and all communities involved in this step.
        """        
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
            self._comm,
            current_comm,
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
                        self._comm,
                        comm,
                        vertex_id,
                        community_edges[comm],
                        degree,
                    )
                local_options[comm] = dq + cost_of_leaving
                set_names.append(comm)
                neighbor_comms[comm] = frozenset(neighbor_comm)

        return local_options, next_comms
 
    def cross_intersect(self, vertex1, comm1, vertex2, comm2):
        """For each pair of 1 and 2 find the intersection and then add the cross products.

        Args:
            vertex1 ((frozen)set[int]): Set of vertices in vertex 1.
            comm1 ((frozen)set[int]): Set of vertices in community 1.
            vertex2 ((frozen)set[int]): Set of vertices in vertex 2.
            comm2 ((frozen)set[int]): Set of vertices in community 2.

        Returns:
            int: Cross intersection.
        """        
        vv = len(vertex1.intersection(vertex2))
        vc = len(vertex1.intersection(comm2))
        cv = len(comm1.intersection(vertex2))
        cc = len(comm1.intersection(comm2))
        return vv*cc + vc*cv
           
    def calculate_interactions(self, set_list1, set_list2, normalisation_factor):
        """Calulates the interaction between each pair of possible moves due to the consistency.

        Args:
            set_list1 (namedtuple[set,set,list[set],list[int]]): Tuple containing all of the sets of vertices involved in the moves in the first graph.
            set_list2 (namedtuple[set,set,list[set],list[int]]): Tuple containing all of the sets of vertices involved in the moves in the second graph.
            normalisation_factor (float): Normalisation factor based on the number of vertices in the graphs.

        Returns:
            dict[int:dict[int:float]]: Nested dictionary containing all of the interactions with the target communities as the keys of the dicts.
        """        
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
        """Combines the possible moves so far with the possible moves in the next graph to find new possible moves.

        Args:
            final_options (list[namedtuple]): List of the best overall moves, one for each move in the base graph.
            intermediate_options (list[namedtuple]): List of all of the best moves for each pair of base move and move in the previous graph.
            local_options (dict[int:tuple]): Dict of all of the moves in the current graph, whith the change in consistency.
            interactions (dict[int:dict[int:float]]): Nested dict of interactions between moves in the previous graph and moves in the current graph.
            graph_id (int): Index of the current graph.

        Returns:
            list[namedtuple]: List of all of the best moves for each pair of base move and move in the current graph.
        """        
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
        """Performs the local move step of the consistency leiden algorithm 2 for one vertex.

        Args:
            graph_id (int): Index of the current graph.
            vertex_id (int): Index of the current vertex.
            queue (simplequeue): Queue of vertices into which newly queued vertices can be put.
        """        
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
        
        for comm in local_options:
            dq = sum(local_options[comm][0::2])
            dc = sum(local_options[comm][1::2])
            if dq>0 or dc>0:
                final_options.append(final_option(comm, {graph_id:comm}, sum(local_options[comm])))
        if len(final_options) == 0:
            return
            
        for direction, limit in [(-1, -1), (1, len(self.graph_stack[0]))]:
            previous_comms = base_comms
            previous_graph_id = graph_id
            vertex_id = selected_vertices[graph_id]
            options = [option(opt.source_target, graph_id, opt.source_target, opt.target_comms, opt.dQ) for opt in final_options]
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
        for graph_id in final_option.target_comms:
            graph = graphs[graph_id]
            vertex_id = selected_vertices[graph_id]
            target_comm = final_option.target_comms[graph_id]
            
            if target_comm == -1:
                i = 0
                while True:
                    if communities.get(i, (0, 0, 0, -1, True))[0] == 0:
                        communities.pop(i, None)
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
                        graph.vs[vertex][self._inQueue] = True

            graph.vs[vertex_id][self._queued] = False
       
    def refine(self):
        """Performs the refinement step of the consistency leiden algorithm 2.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """        
        self.converged = True
        graphs = self.graph_stack[-1]
        for graph_id, graph in enumerate(graphs):
            graph.vs[self._queued] = True
        communities = list(self.communities[self._comm].keys())
        refine_communities = self.communities[self._refine]
        random.shuffle(communities)
        for comm in communities:
            _, _, degreesum, graph_id, _ = self.communities[self._comm][comm]
            graph = graphs[graph_id]
            kwarg = {self._comm + "_eq": comm}
            indices = [v.index for v in graph.vs.select(**kwarg)]
            random.shuffle(indices)
            
            # Check which vertices are wellconnected
            for vertex_id in indices:
                neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
                degree = graph.vs[vertex_id][self._degree]
                edges = 0
                for neighbor in neighbors:
                    edges += graph[vertex_id, neighbor.index]
                wellconnected = edges >= degree*(degreesum-degree)/(graph[self._m]*2)
                graph.vs[vertex_id][self._wellConnected] = wellconnected
                a, b, c, d, _ = refine_communities[graph.vs[vertex_id][self._refine]]
                refine_communities[graph.vs[vertex_id][self._refine]] = (a, b, c, d, wellconnected)

            for vertex_id in indices:
                if not graph.vs[vertex_id][self._queued]:
                    continue
                if not graph.vs[vertex_id][self._wellConnected]:
                    continue
                if self.refineVertex(graph, vertex_id, graph_id, kwarg):
                    
                    target = graph.vs[vertex_id[self._refine]]
                    kwarg2 = {self._refine + "_eq": target}
                    members = [v.index for v in graph.vs[indices].select(**kwarg2)]
                    edges = 0
                    for member in members:
                        neighbors = graph.vs[graph.neighbors(member)].select(**kwarg)
                        for neighbor in neighbors:
                            edges += graph[vertex_id, neighbor.index]
                    ref_mult, ref_edges, ref_degreesum, graph_id, _ = refine_communities[target]
                    if not (edges-2*ref_edges) >= ref_degreesum*(degreesum-ref_degreesum)/(graph[self._m]*2):
                        refine_communities[target] = (ref_mult, ref_edges, ref_degreesum, graph_id, False)
        return self
    
    def refineVertex(self, graph, vertex_id, graph_id, kwarg):
        """Performs the refinement step of the consistency leiden algorithm 2 for one vertex.

        Args:
            graph (ig.Graph): Current graph.
            vertex_id (int): Index of current vertex.
            graph_id (int): Index of current graph.
            kwarg (dict): Dictionary containing the argument for the select function call that selects only members of the same community as the current vertex.
        """        
        neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
        degree = graph.vs[vertex_id][self._degree]
        current_comm = graph.vs[vertex_id][self._refine]
        self_edges = graph.vs[vertex_id][self._selfEdges]
        refine_communities = self.communities[self._refine]

        community_edges = {}
        neighbor = {}
        for vertex in neighbors:
            refine_comm = vertex[self._refine]
            if not refine_communities[refine_comm][4]:
                continue
            community_edges[refine_comm] = (
                community_edges.get(refine_comm, 0)
                + graph[vertex_id, vertex.index]
            )
            neighbor[refine_comm] = vertex

        candidates = []
        weights = []
        cost_of_leaving, _, _ = self.calculateDQMinus(
            self._refine,
            self._refineConsistency,
            current_comm,
            vertex_id,
            0,
            degree,
        )
        for refine_comm in community_edges:
            dq, _ = self.calculateDQPlus(
                    self._refine,
                    self._refineConsistency,
                    refine_comm,
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
            return True
        return False

    def update_communities(self, attr, current, future, community_edges, multiplicity, self_edges, degree):
        """Update the community dictionary to reflect the changes caused by a move.

        Args:
            attr (string): Attribute in which the move was made (self._comm or self._refine).
            current (int): Community that the vertex is leaving.
            future (int): Community that the vertex is joining.
            community_edges (dict[int:int]): summary of the number of edges between the vertex and all communities.
            multiplicity (int): Number of vertices represented by the moved vertex.
            self_edges (int): Number of edges between vertices represented by the moved vertex.
            degree (int): Total degree of the vertices represented by the moved vertex.
        """        
        communities = self.communities[attr]
        old_vertexcount, old_edgecount, old_degreesum, graph_id, _ = communities[current]
        communities[current] = (
            old_vertexcount - multiplicity,
            old_edgecount - (community_edges.get(current, 0) + self_edges),
            old_degreesum - degree,
            graph_id,
            True,
        )
        old_vertexcount, old_edgecount, old_degreesum, graph_id, _ = communities.get(
            future, (0, 0, 0, graph_id, True)
        )
        communities[future] = (
            old_vertexcount + multiplicity,
            old_edgecount + (community_edges.get(future, 0) + self_edges),
            old_degreesum + degree,
            graph_id,
            True,
        )
        
    def deAggregateVertex(self, graph_id, *vertices):
        """Translate each list of vertices in the top level graph into a list of vertices in the bottom level graph.

        Args:
            graph_id (int): Index of the graph.

        Returns:
            tuple[list[int]]: For each input list a list of indices.
        """        
        for graphs in self.graph_stack[:0:-1]:
            subvertices = graphs[graph_id].vs[self._subVertices]
            vertices = tuple(sum([subvertices[v] for v in item], start=[]) 
                             for item in vertices)
        return vertices
        
    def reAggregateVertex(self, graph_id, *vertices):
        """Translate each list of vertices in the bottom layer graph into a list of vertices in the top level graph.

        Args:
            graph_id (int): Index of the graph.

        Returns:
            tuple[list[int]]: For each input list a list of indices.
        """        
        for graphs in self.graph_stack[:-1]:
            refineIndices = graphs[graph_id].vs[self._refineIndex]
            vertices = tuple([refineIndices[v] for v in item] 
                             for item in vertices)
        return vertices
            
    def countCommunities(self, graph, attr, *vertices):
        """Translate each list of vertices into a dictionary counting how often each community is found in those vertices.

        Args:
            graph (ig.Graph): The current graph.
            attr (string): The vertex attribute used to get the community.

        Returns:
            list[dict]: For each input list a dict mapping communities to the number of listed vertices in that community.
        """        
        output = []
        lookup = graph.vs[attr]
        for _list in vertices:
            _dict = {}
            for v in _list:
                val = lookup[v]
                _dict[val] = _dict.get(val, 0) + 1
            output.append(_dict)
        return output

    def calculateDQPlus(self, attr, attr_cons, comm, vertex_id, edges, degree, deaggregated_vertex = None):
        """Calculates the changes of modularity and consistency for moving a disconnected vertex into a community.

        Args:
            attr (string): Name of the attribute under consideration (self._comm or self._refine).
            comm (int): Target community to consider.
            vertex_id (int): Index of the current vertex.
            edges (int): Total weight of the edges between the vertex and the community.
            degree (int): Total degree of the vertices within the current vertex.
            deaggregated_vertex (list[int], optional): List of the base vertices represented by the current vertex. Defaults to None.

        Returns:
            tuple: change in modularity, change in consistency.
            list[int]: list of the base vertices in the target community.
        """        
        vertexcount, edgecount, degreesum, graph_id, _ = self.communities[attr][comm]
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
            adj_comm, vertex_comm = self.countCommunities(graph, attr_cons, _comm_members, _vertex)
                
            for key in vertex_comm:
                dConsistency += vertex_comm[key] * adj_comm.get(key, 0) * 2
            dConsistency -= len(vertex) * len(comm_members)
            
        n = self.graph_stack[0][graph_id][self._n]
        dc = 2*dConsistency / (n * (n - 1))

        return (dq, self.consistency_weight * dc), comm_members

    def calculateDQMinus(self, attr, attr_cons, comm, vertex_id, edges, degree):
        """Calculates the changes of modularity and consistency for moving a disconnected vertex out of a community.

        Args:
            attr (string): Name of the attribute under consideration (self._comm or self._refine).
            comm (int): Community being left.
            vertex_id (int): Index of the current vertex.
            edges (int): Total weight of the edges between the vertex and the community.
            degree (int): Total degree of the vertices within the current vertex.

        Returns:
            tuple: change in modularity, change in consistency.
            list[int]: list of the base vertices in the target community.
            list[int]: list of the base vertices in the vertex.
        """        
        vertexcount, edgecount, degreesum, graph_id, _ = self.communities[attr][comm]
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
            adj_comm, vertex_comm = self.countCommunities(graph, attr_cons, _comm_members, _vertex)

            for key in vertex_comm:
                dConsistency += vertex_comm[key] * adj_comm.get(key, 0) * 2
            dConsistency -= len(vertex) * len(comm_members)

        n = self.graph_stack[0][graph_id][self._n]
        dc = -2*dConsistency / (n * (n - 1))

        return (dq, self.consistency_weight * dc), comm_members, vertex

    def graph_neigbors(self, graph_id):
        """Lists the graphs adjacent to the current graph.

        Args:
            graph_id (int): Current graph.

        Returns:
            list[int]: Indeces of adjacent graphs.
        """        
        output = []
        if graph_id > 0:
            output.append(graph_id - 1)
        if graph_id < len(self.graph_stack[0]) - 1:
            output.append(graph_id + 1)
        return output

    def aggregate(self):
        """Creates the next layer of graphs in the graph stack by aggregating vertices with the same self._refine value.

        Returns:
            LeidenClass: Self, for chaining of functions.
        """        
        new_layer = []
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
            new_layer.append(aggregate_graph)
        self.graph_stack.append(new_layer)
        return self

    def deAggregate(self):
        """Removes all added layers of graphs while preserving the self._comm attributes.

        Returns:
            LeidenComm: Self, for chaining of functions.
        """        
        self.deAggregate_Unwind(self._comm)
        self.graph_stack = [self.graph_stack[0]]
        return self

    def deAggregate_Unwind(self, attr):
        """Distributes the community membership from the top layer of graphs to the bottom layer.

        Args:
            attr (string): Name of the community membership to distribute.
        """        
        for graphs, aggregate_graphs in zip(
            self.graph_stack[-2::-1], self.graph_stack[:0:-1]
        ):
            for graph, aggregate_graph in zip(graphs, aggregate_graphs):
                graph.vs[attr] = [
                    aggregate_graph.vs[index][attr]
                    for index in graph.vs[self._refineIndex]
                ]

    def renumber(self, attr):
        """Renumbers the community membership on the bottom layer of graphs, DOESN'T UPDATE MEMBERSHIP DICT.

        Args:
            attr (string): Name of attribute to renumber.

        Returns:
            leidenClass: Self, for chaining of functions.
        """        
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
        """Calculates the current quality, even if graph is still stacked.

        Args:
            attr (string): Name of community structure to calculate the quality of.
            consistency_weight (float): Consistency weight to use.

        Returns:
            tuple[float, float]: Modularity, consistency.
        """        
        self.deAggregate_Unwind(attr)
        return quality(self.graph_stack[0], attr, consistency_weight)


def quality(graphs, attr, consistency_weight):
    """Calculates the quality of a sequence of graphs.

    Args:
        graphs (list[ig.Graph]): List of graphs.
        attr (string): Name of the attribute containing the community structure.
        consistency_weight (float): Relative weight of the consistency.

    Returns:
        tuple[float, float]: Modularity, consistency.
    """    
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
    """Renumbers a list from 0 to k, keeping same values the same

    Args:
        lst (list[hashable]): Input list

    Returns:
        list[int]: Renumbered list
    """    
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

    for i in range(1):
        # graph = ig.Graph.Famous("Cubical")
        # graph = ig.Graph.Famous("zachary")
        random.seed(i)
        graphs = [GrGen.GirvanNewmanBenchmark(
            7, 4 * i, density=0.8) for i in range(10)]
        random.seed(i)
        print(leiden(graphs, "comm", 2, 0.8))
        cluster = ig.VertexClustering.FromAttribute(graphs[0], "comm")
        ig.plot(cluster, "test.pdf")
