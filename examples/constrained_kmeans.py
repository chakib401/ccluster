import matplotlib.pyplot as plt
import numpy as np
from ccluster.size import constrained_k_means

# generate a uniform grid
nx = np.arange(0, 100, 5)
ny = np.arange(0, 100, 5)
x_coords, y_coords = np.meshgrid(nx, ny)

data = np.column_stack([x_coords.reshape(-1), y_coords.reshape(-1)])

n_clusters = 4

constraints = [
    [100, 100, 100, 100],
    [200, 120, 40, 40],
    [260, 120, 10, 10]
]

for cluster_size in constraints:
    _, labels, _ = constrained_k_means(data,
                                       n_clusters=n_clusters,
                                       cluster_sizes=cluster_size)

    # plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('off')
    plt.title(f'{n_clusters} clusters with sizes {*cluster_size,}')
    plt.show()
