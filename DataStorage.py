import os
import numpy as np
import warnings

# V1: "#n_graphs, step_size, k_gen, density, k_cluster, modularity, consistency\n"
# V2: "#n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n"
# Order: "#generator, clusterer, n_graphs, step_size, k_gen, density, seed, k_cluster, iterations\n"


def update(filename):
    if not os.path.isfile(filename):
        return
    with open(filename) as file:
        data = np.genfromtxt(file, dtype=(int, int, float, float, float, float, float), delimiter=",", skip_header=1, names=True)
    with open(filename, 'x') as file:
        file.write("#n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n")
        for line in data:
            file.write(f'{line["n_graphs"]}, {line["step_size"]}, {line["k_gen"]}, {line["density"]}, 0, {line["k_cluster"]}, {line["modularity"]}, {line["consistency"]}, 2\n')
            
def loadData(filename):
    if not os.path.isfile(filename):
        warnings.warn(f"Failed to load file {filename} for plotting")
    with open(filename) as file:
        data = np.genfromtxt(file, dtype=(int, int, float, float, int, float, float, float, int), delimiter=",", skip_header=1, names=True)
    return data

def writeData(filename, clustering_func, generator, n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations):
    if not os.path.isfile(filename):
        with open(filename, "a") as file:
            file.write(f"#Results of {clustering_func.__name__} on graphs generated using {generator.__name__}\n")
            file.write("#n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n")
    with open(filename, "a") as file:
        file.write(f"{n_graphs}, {step_size}, {k_gen}, {density}, {seed}, {k_cluster}, {modularity}, {consistency}, {iterations}\n")