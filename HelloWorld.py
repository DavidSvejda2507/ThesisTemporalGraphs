import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn

ig.config["plotting.backend"] = "matplotlib"


# G = ig.Graph.GRG(100, 0.2)
for i in range(1):
    G1 = grGen.GirvanNewmanBenchmark(6, 2 * i)
    G2 = grGen.GirvanNewmanBenchmark(6, 2 * (i + 1))
    # print(G)

    shapeMap = grPlt.shapeMap()
    shapes1 = [shapeMap[community] for community in G1.vs["community"]]
    shapes2 = [shapeMap[community] for community in G2.vs["community"]]
    # print(shapes)

    part1 = leidenalg.find_partition(G1, leidenalg.ModularityVertexPartition)
    part2 = leidenalg.find_partition(G2, leidenalg.ModularityVertexPartition)
    part = leidenalg.find_partition(
        grGen.mergeGraphs(G1, G2), leidenalg.ModularityVertexPartition
    )
    print(
        f"Accuracy on G1: {grAn.calculateAccuracy(part1.membership, G1.vs['community'])[0]}"
    )
    print(
        f"Accuracy on G2: {grAn.calculateAccuracy(part2.membership, G2.vs['community'])[0]}"
    )
    print(
        f"Accuracy of union on G1: {grAn.calculateAccuracy(part.membership, G1.vs['community'])[0]}"
    )
    print(
        f"Accuracy of union on G1: {grAn.calculateAccuracy(part.membership, G2.vs['community'])[0]}"
    )
    # print(
    #     f"The modularity of the 'correct' clustering is \t{G.modularity(G.vs['community'])}"
    # )
    # print(
    #     f"The modularity of the found clustering is \t{G.modularity(part.membership)}"
    # )

    # ig.plot(
    #     part,
    #     "plotPart_" + str(i) + ".pdf",
    #     vertex_shape=shapes,
    #     **grPlt.defaultVisualStyle(layout="circle"),
    # )

# ig.plot(part, "plotPart.pdf")
# print(part)
# visualStyle = grPlt.defaultVisualStyle()
# colormap = grPlt.colormap()
# visualStyle["vertex_color"] = [colormap[partition] for partition in part.membership]

# ig.plot(G, "plot.pdf", **visualStyle)
