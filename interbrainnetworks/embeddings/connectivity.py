"""Wrapper for connectivity analysis."""
import warnings
import numpy as np
import networkx as nx
from networkx.algorithms.bipartite.matrix import biadjacency_matrix

warnings.filterwarnings(action='ignore', category=FutureWarning)


def _check_consitency(graphs: list) -> list:
    """Check if all graphs have the same number of nodes.

    Args:
        graphs (list): list of networkx graphs.

    Returns:
        list: list of consistent graphs.
    """
    # graphs = np.asarray(graphs)

    all_n_nodes = [g.number_of_nodes() for g in graphs]

    # check equal number of nodes
    if len(set(all_n_nodes)) != 1:

        nodes = {}
        # get all nodes
        for g in graphs:
            nodes.update(g.nodes(data=True))

        # update all graphs
        for g in graphs:
            g.add_nodes_from(
                [(node, attr) for (node, attr) in nodes.items()]
            )

    return graphs


def _check_connectivity_vector(graph: nx.graph,
                               feature: np.ndarray) -> np.ndarray:
    """Check if the feature vector is compatible with the graph weights.

    Args:
        graph (nx.graph): networkx graph.
        feature (np.ndarray): feature vector.

    Raises:

    Returns:
        np.ndarray: connectivity vector.
    """

    edges = nx.to_edgelist(graph)
    weights = [d['weight'] for _, _, d in edges]

    # vectorize connectivity estimates
    if feature.ndim > 1:
        feature = np.reshape(feature, -1)

    if (len(weights) > 0 and
       not np.isclose(np.sum(weights), np.sum(feature))):
        raise ValueError('Sum of weights and feature are not equal.')

    return feature


class Connectivity(object):

    def __init__(self):
        """Initialize the connectivity vector."""
        pass

    def fit(self, graphs: list):
        """Fits the embedding by the edge weights of each graph.

        Args:
            graphs (list): list of netowrkx graphs.
        """

        graphs = _check_consitency(graphs)

        features = []
        for graph in graphs:

            graph = nx.to_undirected(graph)

            # specify set of nodes
            top = [n for n, v in
                   nx.get_node_attributes(graph, 'bipartite').items()
                   if v == 0]
            bottom = [n for n, v in
                      nx.get_node_attributes(graph, 'bipartite').items()
                      if v == 1]
            top.sort()
            bottom.sort()

            # get the connectivity vector of the bipartite graph
            adj = biadjacency_matrix(graph, top, bottom).todense()
            feature = _check_connectivity_vector(graph, adj)

            features.append(feature)

        self._embedding = features
        return self

    def get_embedding(self) -> np.array:
        """Getting the embedding of graphs.

        Returns:
            np.array:  The embedding of graphs.
        """
        result = np.stack(self._embedding, axis=0)
        return result
