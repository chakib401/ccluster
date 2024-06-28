import matplotlib.pyplot as plt
import numpy as np
from ccluster.size import ConstrainedSpectralClustering

# generate a spiral
x_coords = []
y_coords = []

for theta in np.linspace(7, 10 * np.pi, 400):
    r = theta ** 2
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))

data = np.column_stack([x_coords, y_coords])

n_clusters = 4

constraints = [
    [100, 100, 100, 100],
    [200, 120, 40, 40],
    [260, 120, 10, 10]
]

for cluster_size in constraints:
    labels = ConstrainedSpectralClustering(
        n_clusters=n_clusters,
        cluster_sizes=cluster_size,
        affinity='nearest_neighbors',
        n_neighbors=2
    ).fit_predict(data)

    # plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('off')
    plt.title(f'{n_clusters} clusters with sizes {*cluster_size,}')
    plt.show()
