"""Network embedding."""

import numpy as np
from sklearn.decomposition import NMF
from sklearn.utils.validation import check_is_fitted


def _check_consitency(graphs: list) -> list:
    """Check if dictionary is consistent.

    Args:
        topo (list): list of topological metrics.

    Returns:
        list: list of consistent topological metrics and nodes.
    """

    all_nodes = [list(g.keys()) for g in graphs]
    all_nodes = [n for g in all_nodes for n in g]

    list_nodes = sorted(set(all_nodes))

    # check equal number of nodes
    if len(list_nodes) != 1:

        # initialize dictionary with zeros
        nodes = {n: 0 for n in list_nodes}

        for g in graphs:
            query = {n: 0 for n in set(nodes) if n not in g.keys()}
            g.update(query)

    return graphs, list_nodes


class Embedding(object):

    def __init__(self,
                 n_components: int = 5,
                 init: str = 'nndsvd',
                 solver: str = 'cd',
                 tol: float = 1e-4,
                 max_iter: int = 10000,
                 beta_loss: str = 'frobenius',
                 log: bool = True,
                 seed: int = 20221001,
                 **kwargs):
        """Initialize embedding.

        Args:
            n_components (int, optional): Number of components. Defaults to 5.
            init (str, optional): Initialization method. Defaults to 'nndsvd'.
            solver (str, optional): Solver method. Defaults to 'cd'.
            tol (float, optional): Tolerance. Defaults to 1e-4.
            max_iter (int, optional): Maximum number of iterations.
                Defaults to 10000.
            beta_loss (str, optional): Loss function. Defaults to 'frobenius'.
            log (bool, optional): Reurn log scaled values. Defaults to False.
            seed (int, optional): Random seed. Defaults to 20221001.
        """
        self._decomposition = NMF(n_components,
                                  init=init,
                                  solver=solver,
                                  tol=tol,
                                  max_iter=max_iter,
                                  beta_loss=beta_loss,
                                  random_state=seed,
                                  **kwargs)
        self._log = log

    @property
    def basis(self):
        """Getting the basis of the decomposition."""
        basis = None
        if (hasattr(self, '_embedding') and
           check_is_fitted(self._decomposition)):
            basis = self._decomposition.components_
        return basis

    def fit(self, graphs: list):
        """Fits the embedding by the edge weights of each graph.

        Args:
            graphs (list): list of netowrkx graphs.
        """

        graphs, nodes = _check_consitency(graphs)

        features = []
        for graph in graphs:
            feature = [v for k, v in sorted(graph.items())]
            features.append(feature)

        features = np.stack(features, axis=0)

        # add node in / node ou
        if np.amin(features) < 0:
            positive = np.abs(features) * np.sign(np.sign(features) + 1)

            negative = np.abs(features) * -1 * np.sign(np.sign(features) - 1)
            features = np.concatenate((positive, negative), axis=1)

        features = self._decomposition.fit_transform(features)
        self._nodes = nodes
        self._embedding = features
        return self

    def get_embedding(self) -> np.array:
        """Getting the embedding of graphs.

        Returns:
            np.array:  The embedding of graphs.
        """

        result = self._embedding
        if self._log:
            result = np.log1p(result)

        return result

    def get_factorization(self) -> np.array:
        """Getting the decomposition basis matrix.

        Returns:
            np.array:  The basis matrix of the graph embedding.
        """
        return self._decomposition.components_
