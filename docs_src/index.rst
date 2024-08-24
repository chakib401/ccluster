ccluster: a package for clustering with size constraints
=========================================================

*ccluster* is a library for performing clustering with exhaustive or
partial cluster size constraints. It currently provides two constrained clustering
algorithms: a constrained *k*-means suitable for euclidean data, and a
constrained graph clustering algorithm based on spectral clustering.

+-----------------------+-----------------------+----------------------+
| |k-means example 1|   | |k-means example 2|   | |k-means example 3|  |
+=======================+=======================+======================+
| |spectral clustering  | |spectral clustering  | |spectral clustering |
| example 1|            | example 2|            | example 3|           |
+-----------------------+-----------------------+----------------------+


.. |k-means example 1| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km1.png
.. |k-means example 2| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km2.png
.. |k-means example 3| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/km3.png
.. |spectral clustering example 1| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc1.png
.. |spectral clustering example 2| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc2.png
.. |spectral clustering example 3| image:: https://raw.githubusercontent.com/chakib401/ccluster/main/docs_src/_static/sc3.png


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   intro.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: API Documentation

   api/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Examples

   examples/index
