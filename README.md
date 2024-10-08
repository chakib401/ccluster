# ccluster: a package for clustering with size constraints

[![PyPI version](https://badge.fury.io/py/ccluster.svg)](https://pypi.org/project/ccluster) 
![Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-4AC51C)
[![Run Status](https://github.com/chakib401/ccluster/actions/workflows/python-test.yml/badge.svg)](https://github.com/chakib401/ccluster/actions) 
[![Docs Status](https://readthedocs.org/projects/ccluster/badge/?version=latest)](https://ccluster.readthedocs.io/)

*ccluster* is a library for performing clustering with exhaustive or partial cluster size constraints. 
It currently provides two constrained clustering algorithms: a constrained *k*-means suitable for euclidean data, and a constrained graph clustering algorithm based on spectral clustering.


For more details, please refer to the [documentation](https://ccluster.readthedocs.io/). 

| ![k-means example 1](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km1.png)             | ![k-means example 2](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km2.png)             | ![k-means example 3](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km3.png)            |
|----------------------------------------------------------|----------------------------------------------------------|---------------------------------------------------------|
| ![spectral clustering example 1](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc1.png) | ![spectral clustering example 2](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc2.png) | ![spectral clustering example 3](https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc3.png) |

### Installation

This package is available on Pypi. It can be installed as follows:
```bash
pip install ccluster
```

A set of dependencies will also be installed. The package can now be used:

```python
import ccluster
```

### Quick-Start

Here is an example of using constrained $k$-means with exhaustive constraints i.e. giving a size constraint for each cluster:

 ```python
>>> from ccluster.size import ConstrainedKMeans
>>> import numpy as np

>>> X = np.array([[1, 2], [1, 4], [1, 0],
...               [10, 2], [10, 4], [10, 0]])
>>> kmeans = ConstrainedKMeans(
...     n_clusters=2,
...     cluster_size=[2, 4],
...     random_state=0,
...     n_init=10).fit(X)
>>> kmeans.labels_
# array([1, 1, 1, 0, 0, 1])  
>>> kmeans.predict([[0, 0], [12, 3]], [1, 1])
# array([1, 0], dtype=int32)  
>>> kmeans.cluster_centers_
# array([[10. , 3. ],  
#        [ 3.25, 1.5 ]])  
```  

It is also possible to specify partial constraints i.e. constraints on a subset of the clusters and leave the remaining ones free. Here is an example using constrained spectral clustering:

 ```python
>>> from ccluster.size import ConstrainedSpectralClustering
>>> import numpy as np

>>> X = np.array([[1, 1], [2, 1], [1, 0],
...               [4, 7], [3, 5], [3, 6],
...               [9, 6], [5, 4], [2, 1]])

>>> spectral = ConstrainedSpectralClustering(
...     n_clusters=4, 
...     cluster_sizes=[2, 2],
...     random_state=0).fit(X)

>>> spectral.labels_
# array([2, 3, 2, 0, 3, 0, 1, 1, 3])

```

