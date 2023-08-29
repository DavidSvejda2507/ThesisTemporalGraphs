import igraph as ig
import leidenalg
import matplotlib.pyplot as plt
import GraphGenerators as gen
import GraphPlotting as grPlt

ig.config["plotting.backend"] = "matplotlib"


# G = ig.Graph.GRG(100, 0.2)
for i in range(8):
    G = gen.GirvanNewmanBenchmark(i)
    # print(G)

    shapeMap = grPlt.shapeMap()
    shapes = [shapeMap[community] for community in G.vs["community"]]
    # print(shapes)

    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition)

    ig.plot(
        part,
        "plotPart_" + str(i) + ".pdf",
        vertex_shape=shapes,
        **grPlt.defaultVisualStyle()
    )

# ig.plot(part, "plotPart.pdf")
# print(part)
# visualStyle = grPlt.defaultVisualStyle()
# colormap = grPlt.colormap()
# visualStyle["vertex_color"] = [colormap[partition] for partition in part.membership]

# ig.plot(G, "plot.pdf", **visualStyle)
