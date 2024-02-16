import os
import numpy as np
import warnings
import icecream as ic

# V1: "#n_graphs, step_size, k_gen, density, k_cluster, modularity, consistency\n"
# V2: "#n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n"
# Order: "#generator, clusterer, n_graphs, step_size, k_gen, density, seed, k_cluster, iterations\n"


def update(filename):
    if not os.path.isfile(filename):
        warnings.warn(f"Failed to load file {filename}")
        return
    try:
        with open(filename, 'r') as file:
            header = file.readline()
            data = np.genfromtxt(file, dtype=(int, int, float, float, float, float, float), delimiter=",", skip_header=0, names=True)
        
        with open(filename, 'w') as file:
            file.write(header)
            file.write("n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n")
            for line in data:
                file.write(f'{line["n_graphs"]}, {line["step_size"]}, {line["k_gen"]}, {line["density"]}, 0, {line["k_cluster"]}, {line["modularity"]}, {line["consistency"]}, 2\n')
    except(ValueError):
        pass
            
def loadData(filename):
    if not os.path.isfile(filename):
        warnings.warn(f"Failed to load file {filename}")
        return None
    with open(filename, 'r') as file:
        data = np.genfromtxt(file, dtype=(int, int, float, float, int, float, float, float, int), delimiter=",", skip_header=1, names=True)
    return data

def writeData(filename, clustering_func, generator, n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations):
    if not os.path.isfile(filename):
        with open(filename, "a") as file:
            file.write(f"Results of {clustering_func.__name__} on graphs generated using {generator.__name__}\n")
            file.write("n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations\n")
    with open(filename, "a") as file:
        file.write(f"{n_graphs}, {step_size}, {k_gen}, {density}, {seed}, {k_cluster}, {modularity}, {consistency}, {iterations}\n")
        
def maybeWriteData(filename, clustering_func, generator, n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations):
    data = loadData(filename)
    for l in data:
        if all([
            l["n_graphs"] == n_graphs,
            l["step_size"] == step_size, 
            l["k_gen"] == k_gen, 
            l["density"] == density, 
            l["seed"] == seed, 
            l["k_cluster"] == k_cluster, 
            l["iterations"] == iterations,
        ]): return
    writeData(filename, clustering_func, generator, n_graphs, step_size, k_gen, density, seed, k_cluster, modularity, consistency, iterations)
    