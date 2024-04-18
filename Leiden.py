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
_wellconnected = "leiden_wellconnected"
_multiplicity = "leiden_multiplicity"
_degree = "leiden_degree"
_selfEdges = "leiden_selfEdges"
_m = "leiden_m"

theta = 1e-2


def leiden(graph, attr, iterations, simplified = False):
    initialiseGraph(graph)
    communities = initialisePartition(graph, _comm)

    for _ in range(iterations):
        localMove(graph, communities)
        communities = cleanCommunities(communities)

        refine_communities = initialisePartition(graph, _refine)
        converged = refine(graph, communities, refine_communities, simplified)
        refine_communities = cleanCommunities(refine_communities)

        graphs = [graph]

        while not converged:
            graph = aggregate(graph, refine_communities)
            graphs.append(graph)

            localMove(graph, communities)
            communities = cleanCommunities(communities)

            converged = refine(graph, communities, refine_communities, simplified)
            refine_communities = cleanCommunities(refine_communities)

        for graph, aggregate_graph in zip(graphs[-2::-1], graphs[:0:-1]):
            deAggregate(graph, aggregate_graph)

        graph = graphs[0]

    graph.vs[attr] = graph.vs[_comm]
    del graph.vs[_comm]
    del graph.vs[_refine]
    del graph.vs[_refineIndex]
    del graph.vs[_queued]
    if not simplified:
        del graph.vs[_wellconnected]
    del graph.vs[_degree]
    del graph.vs[_selfEdges]
    del graph[_m]

    renumber(graph, attr)

    return quality(graph, attr)


def quality(graph, comm):
    return graph.modularity(graph.vs[comm])


def initialiseGraph(graph):
    """Add weights if they are not present yet,
    Store the degree of each vertex in each vertex,
    Initialise selfedges and multiplicity,
    Store the number of edges in m

    Args:
        graph (ig.Graph): Graph to apply the algorithm to
    """
    m2 = 0
    if graph.is_weighted():
        for vertex in graph.vs:
            degree = 0
            for neighbor in vertex.neighbors():
                degree += graph[vertex.index, neighbor]
            vertex[_degree] = degree
            m2 += degree
    else:
        graph.es["weight"] = 1
        graph.vs[_degree] = graph.degree(range(graph.vcount()))
        m2 = sum(graph.vs[_degree])

    graph.vs[_selfEdges] = 0
    graph.vs[_multiplicity] = 1
    graph[_m] = m2 / 2


def initialisePartition(graph, attribute):
    """Initialise a partition by assigning each vertex a different index,
    And make a dictionary containing the number of vertices, internal edges and total degree of each community

    Args:
        graph (ig.Graph): Graph to initialise the partition of
        attribute (string): vertex attribute to store the indices in

    Returns:
        dict: Dictionary containing summaries of all of the communities and the empty community
    """
    communities = {}
    for index, vertex in enumerate(graph.vs):
        vertex[attribute] = index
        communities[index] = (vertex[_multiplicity], 0, vertex[_degree], True)
    # Moving to community -1 corresponds to moving to a new community
    communities[-1] = (0, 0, 0, True)
    return communities


def cleanCommunities(communities):
    """Make a copy without any empty communities from a dictionary of community summaries

    Args:
        communities (dict): Dictionary of community summaries

    Raises:
        ValueError: Raises an error is a community contains no vertices but still has internal edges or internal degree

    Returns:
        dict: Clean copy of the input dictionary
    """
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
    output[-1] = (0, 0, 0, True)  # Empty community
    return output


def makeQueue(count):
    """Make a queue with the indeces from 0 to count in a random order

    Args:
        count (int): Number of indices to include

    Returns:
        SimpleQueue: Queue object filled with indices
    """
    queue = SimpleQueue()
    indices = list(range(count))
    random.shuffle(indices)
    for i in indices:
        queue.put(i)
    return queue


def localMove(graph, communities):
    """Perform the local move step of the Leiden algorithm

    Args:
        graph (ig.Graph): Graph to work on
        communities (dict): Dictionary containing summaries of the communities that the vertices are members of
    """
    queue = makeQueue(graph.vcount())
    graph.vs[_queued] = True

    while not queue.empty():
        vertex_id = queue.get()
        degree = graph.vs[vertex_id][_degree]
        current_comm = graph.vs[vertex_id][_comm]
        self_edges = graph.vs[vertex_id][_selfEdges]
        neighbors = graph.neighbors(vertex_id)

        # Community_edges contains the total weight of the edges that would be added to or removed from a community if the vertex is moved to/form each community
        community_edges = {-1: self_edges}
        # Include moving to the empty community (-1) as an option
        for vertex in neighbors:
            comm = graph.vs[vertex][_comm]
            community_edges[comm] = (
                community_edges.get(comm, self_edges) + graph[vertex_id, vertex]
            )

        max_dq = 0  # dq is 0 for staying in the current community
        max_comm = current_comm
        cost_of_leaving = calculateDQMinus(
            graph,
            communities,
            current_comm,
            vertex_id,
            community_edges.get(current_comm, self_edges),
            degree,
        )
        for comm in community_edges:
            if comm != current_comm:
                # calculateDQPlus assumes that the vertex is not currently part of the target community
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
            if max_comm == -1:
                i = 0
                while True:
                    if communities.get(i, (0, 0, 0, True))[0] == 0:
                        break  # find a community that is empty or not included in the dictionary
                    i += 1
                max_comm = i

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

            for vertex in neighbors:
                if (
                    not graph.vs[vertex][_queued]
                    and graph.vs[vertex][_comm] != max_comm
                ):
                    graph.vs[vertex][_queued] = True
                    queue.put(vertex)

        graph.vs[vertex_id][_queued] = False


def refine(graph, communities, refine_communities, simplified):
    """Perform the local move step of the leiden algorithm

    Args:
        graph (ig.Graph): Graph to work on
        communities (dict): Dictionary of summaries of the communities
        refine_communities (dict): Dictionary of summaries of the refinement communities
        simplifief (bool): If true disable the wellconnected check and the randomness in choosing the target community

    Returns:
        bool: Whether the refinement step has converged
    """
    converged = True
    # Not actually queued, but we might as well reuse the existing value
    graph.vs[_queued] = True

    for comm in communities:
        # ic(comm)
        # Make a list of all of the vertices in community comm
        kwarg = {_comm + "_eq": comm}
        indices = [v.index for v in graph.vs.select(**kwarg)]
        degreesum = communities[comm][2]
        random.shuffle(indices)
        # ic(indices)

        if not simplified:
            # Check which vertices are wellconnected
            for vertex_id in indices:
                neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
                degree = graph.vs[vertex_id][_degree]
                edges = 0
                for neighbor in neighbors:
                    edges += graph[vertex_id, neighbor.index]
                wellconnected = edges >= degree*(degreesum-degree)/(graph[_m]*2)
                graph.vs[vertex_id][_wellconnected] = wellconnected
                a, b, c, _ = refine_communities[graph.vs[vertex_id][_refine]]
                refine_communities[graph.vs[vertex_id][_refine]] = (a, b, c, wellconnected)
            
        # We don't need to make a queue becuase we only consider vertices that have not yet been merged
        for vertex_id in indices:
            if not graph.vs[vertex_id][_queued]:
                continue
            if (not simplified) and (not graph.vs[vertex_id][_wellconnected]):
                continue
            
            neighbors = graph.vs[graph.neighbors(vertex_id)].select(**kwarg)
            # We only consider neighbors in the same community
            degree = graph.vs[vertex_id][_degree]
            current_comm = graph.vs[vertex_id][_refine]
            self_edges = graph.vs[vertex_id][_selfEdges]

            community_edges = {}
            neighbor = {}
            for vertex in neighbors:
                refine_comm = vertex[_refine]
                if (not simplified) and (not refine_communities[refine_comm][3]):
                    continue
                community_edges[refine_comm] = (
                    community_edges.get(refine_comm, self_edges)
                    + graph[vertex_id, vertex.index]
                )
                neighbor[refine_comm] = vertex
                # The neighbor dict only stores the last neighbor in the list,
                # but that's fine because if there are multiple neigbors in the same community
                # they should all be unqueued already anyway

            candidates = []
            weights = []
            cost_of_leaving = calculateDQMinus(
                graph,
                refine_communities,
                current_comm,
                vertex_id,
                self_edges,  # There should not be any other vertices already in the same community
                degree,
            )
            for refine_comm in community_edges:
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
                if simplified:
                    max_ = max(weights)
                    target = candidates[weights.index(max_)]
                else:
                    target = random.choices(candidates, weights)[0]

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

                neighbor[target][_queued] = False
                graph.vs[vertex_id][_queued] = False
                converged = False
                
                # # Check if the new community is wellconnected
                if not simplified:
                    kwarg2 = {_refine + "_eq": target}
                    members = [v.index for v in graph.vs[indices].select(**kwarg2)]
                    edges = 0
                    for member in members:
                        neighbors = graph.vs[graph.neighbors(member)].select(**kwarg)
                        for neighbor in neighbors:
                            edges += graph[vertex_id, neighbor.index]
                    ref_mult, ref_edges, ref_degreesum, _ = refine_communities[target]
                    if not (edges-2*ref_edges) >= ref_degreesum*(degreesum-ref_degreesum)/(graph[_m]*2):
                        refine_communities[target] = (ref_mult, ref_edges, ref_degreesum, False)
                
        # ic(graph.vs[_refine])
        # input()
    return converged


def update_communities(
    communities, current, future, community_edges, multiplicity, self_edges, degree
):
    """Update the cummunities dictionary to reflect a vertex being moved

    Args:
        communities (dict): Community summaries
        current (int): Community that the vertex moved out of
        future (int): Community that the vertex moves to
        community_edges (dict): Dictionary of total edgeweight between the vertex being moved and vertices in the community
        multiplicity (int): Number of vertices represented by the vertex being moved
        self_edges (double): Total weight of the edges within the vertex being moved
        degree (double): Total degree of the vertices represented by the vertex being moved
    """
    old_vertexcount, old_edgecount, old_degreesum, _ = communities[current]
    communities[current] = (
        old_vertexcount - multiplicity,
        old_edgecount - community_edges.get(current, self_edges),
        old_degreesum - degree,
        True,
    )
    old_vertexcount, old_edgecount, old_degreesum, _ = communities.get(future, (0, 0, 0, True))
    communities[future] = (
        old_vertexcount + multiplicity,
        old_edgecount + community_edges.get(future, self_edges),
        old_degreesum + degree,
        True,
    )


"""
modularity of a community =
edge_count/m - (degree_sum/2m)^2
We want to compare having the vertex in the community vs having it in its own community
"""


def calculateDQPlus(graph, communities, comm, vertex, edges, degree):
    """Calculate the change in modularity if the vertex joins the community

    Args:
        graph (ig.Graph): Current graph
        communities (dict): Community summaries
        comm (int): Index of the community we consider moving to
        vertex (int): Index of the vertex
        edges (double): Total weight of the edges within the vertex and between the vertex and the community
        degree (double): Total degree of the vertex

    Returns:
        double: Change in modularity
    """
    _, _, degreesum, _ = communities[comm]
    dq = edges / graph[_m] - (2 * degreesum * degree) / (2 * graph[_m]) ** 2
    return dq


def calculateDQMinus(graph, communities, comm, vertex, edges, degree):
    """Calculate the change in modularity if the vertex leaves the community

    Args:
        graph (ig.Graph): Current graph
        communities (dict): Community summaries
        comm (int): Index of the community the vertex is currently part of
        vertex (int): Index of the vertex
        edges (double): Total weight of the edges within the vertex and between the vertex and the rest of the community
        degree (double): Total degree of the vertex

    Returns:
        double: Change in modularity
    """
    _, _, degreesum, _ = communities[comm]
    dq = -edges / graph[_m] + (2 * (degreesum - degree) * degree) / (2 * graph[_m]) ** 2
    return dq


def aggregate(graph, communities):
    """Make the aggregate graph where all refinement communities are represented by individual vertices

    Args:
        graph (ig.Graph): Current graph
        communities (dict): Dictionary containing the summaries of the refinement communities

    Returns:
        ig.Graph: Aggregated graph
    """
    partition = ig.VertexClustering.FromAttribute(graph, _refine)
    aggregate_graph = partition.cluster_graph(
        {None: "first", _multiplicity: "sum", _degree: "sum"}, "sum"
    )  # Vertex attributes should be the same between all of the combined vertices, other than multiplicity and degree which are summed
    del partition

    _dict = {v[_refine]: v.index for v in aggregate_graph.vs}
    # refineIndex is a 'pointer' to the index in the aggregated graph which represents that vertex
    graph.vs[_refineIndex] = [_dict[ref] for ref in graph.vs[_refine]]
    aggregate_graph.vs[_selfEdges] = [
        communities[v[_refine]][1] for v in aggregate_graph.vs
    ]

    return aggregate_graph


def deAggregate(graph, aggregate_graph):
    """Deaggregate the community structure of the aggegrated graph to the other graph

    Args:
        graph (ig.Graph): Graph
        aggregate_graph (ig.Graph): Aggregated graph
    """
    graph.vs[_comm] = [
        aggregate_graph.vs[index][_comm] for index in graph.vs[_refineIndex]
    ]


def renumber(graph, comm):
    """Renumbers the attribute to numbers from 0 to n

    Args:
        graph (ig.Graph): Graph
        comm (string): Graph attribute to renumber
    """
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
    for i in range(1):
        # graph = ig.Graph.Famous("zachary")
        random.seed(888)
        graph = GrGen.GirvanNewmanBenchmark(7, density=1)
        random.seed(888)
        print(leiden(graph, "comm", 10))
        cluster = ig.VertexClustering.FromAttribute(graph, "comm")
        ig.plot(cluster, "test.pdf")
