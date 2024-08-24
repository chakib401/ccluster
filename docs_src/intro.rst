Installation
~~~~~~~~~~~~

This package is available on Pypi. It can be installed as follows:

.. code:: bash

   pip install ccluster

A set of dependencies will also be installed. It can also be installed directly from the source code which can be found on the
`project's github page <https://github.com/chakib401/ccluster>`_.


Quick-start
~~~~~~~~~~~

Here is an example of using constrained :math:`k`-means with exhaustive
constraints i.e. giving a size constraint for each cluster:

.. code:: python

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

It is also possible to specify partial constraints i.e. constraints on a
subset of the clusters and leave the remaining ones free. Here is an
example using constrained spectral clustering:

.. code:: python

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
