import matplotlib.pyplot as plt
import numpy as np

methods = {
    0: "leidenalg",
    1: "Python implementation",
    2: "Simplified Python implementation",
    3: "Consistency Leiden 1",
    4: "Consistency Leiden 2",
}

data = np.zeros((5,10000), float)
    
for method in methods:
    for seed in range(100):
        filename = f"ValidationData/{method}_{seed*100}.npy"
        temp = np.load(filename)
        data[method][seed*100:(seed+1)*100] = temp

# colors = ["red", "lime", "blue"]
labels = np.array([methods[i] for i in range(5)])
mask1 = [True, True, True, False, False]
mask2 = [True, False, False, True, True]


for mask, filename in [(mask1,"Leiden_comp.pdf"), (mask2,"Leiden_comp_const.pdf")]:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(data[mask].T, 15, histtype="bar", label=labels[mask])
    ax.legend(prop={"size": 10})
    title = "Histogram of modularities on the Grivan & Newman benchmark"
    ax.set_title(title)
    ax.set_xlabel("Modularity")
    ax.set_ylabel("Count")

    fig.savefig(filename)
    plt.close()
