import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as grGen
import GraphPlotting as grPlt
import GraphAnalysis as grAn

ig.config["plotting.backend"] = "matplotlib"


# G = ig.Graph.GRG(100, 0.2)
for i in range(10):
    G = grGen.GirvanNewmanBenchmark(i)
    # print(G)

    shapeMap = grPlt.shapeMap()
    shapes = [shapeMap[community] for community in G.vs["community"]]
    # print(shapes)

    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)
    print(grAn.calculateAccuracy(part.membership, G.vs["community"]))
    print(
        f"The modularity of the 'correct' clustering is \t{G.modularity(G.vs['community'])}"
    )
    print(
        f"The modularity of the found clustering is \t{G.modularity(part.membership)}"
    )

    ig.plot(
        part,
        "plotPart_" + str(i) + ".pdf",
        vertex_shape=shapes,
        **grPlt.defaultVisualStyle(),
    )

# ig.plot(part, "plotPart.pdf")
# print(part)
# visualStyle = grPlt.defaultVisualStyle()
# colormap = grPlt.colormap()
# visualStyle["vertex_color"] = [colormap[partition] for partition in part.membership]

# ig.plot(G, "plot.pdf", **visualStyle)
