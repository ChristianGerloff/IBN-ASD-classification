"""Wrapper network embeddings."""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from logging import warning
from pathlib import Path
from sklearn import preprocessing
from nilearn.plotting import view_markers
from sklearn.preprocessing import MinMaxScaler

from sklearn.base import BaseEstimator, TransformerMixin


def _check_array(graphs, idx) -> list:
    """Check ensures input is an array.

    Args:
        graphs (_type_): iterrable type of graph information
        idx (_type_): index of graphs

    Returns:
        list: array of graphs, labels, and indices.
    """
    graphs = np.asarray(graphs)
    idx = np.asarray(idx)

    if (graphs.shape != idx.shape):
        raise ValueError('Inputs must be arrays of the same shape. '
                         f'{graphs.shape} != {idx.shape}')
    return graphs, idx


def _check_isolated_nodes(graphs: list) -> list:
    """Checks and removes if graphs have isolated nodes.

    Args:
        graphs (list): list of networkx graphs.

    Returns:
        list: graphs and idx with isolated nodes removed.
    """
    isolated_idx = []
    for idx, graph in enumerate(graphs):
        if nx.number_of_isolates(graph) > 0:
            graph.remove_nodes_from(list(nx.isolates(graph)))
            isolated_idx.append(idx)
    return graphs, isolated_idx


def _feasable_graph(graph: nx.Graph,
                    weakly_connected: bool = False) -> nx.Graph:
    """Ensures graph structure is feasable.

    Args:
        graph (nx.Graph): networkx graph.
        weakly_connected (bool, optional): ensure graph is weakly connected.
            Defaults to False.

    Returns:
        nx.Graph: feasable graph.
    """
    degree = sum(n[1] for n in list(nx.degree(graph)))

    if degree == 0:
        return None
    if weakly_connected:
        graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def _feasable_dict(graph: dict, **kwargs) -> dict:
    """Ensures graph metric is feasable.

    Args:
        graph (dict): graph metric.

    Returns:
        dict: feasable graphs.
    """
    n_values = len(graph)

    if n_values == 0:
        return None
    else:
        return graph


def _check_consitency(graphs: list, weakly_connected: bool = False) -> list:
    """Check if graphs are consistent with embedding assumptions.

    Args:
        graphs (list): list of networkx graphs.
        weakly_connected (bool, optional): ensure graph is weakly connected.
            Defaults to False.

    Returns:
        list: consisten graphs and list of graphs to remove.
    """    ""

    if (all(isinstance(d, nx.Graph) for d in graphs)):
        check = _feasable_graph
    elif (all(isinstance(d, dict) for d in graphs)):
        check = _feasable_dict
    else:
        raise ValueError('Graphs must be either networkx graphs or dicts.')

    pop = []
    for idx, graph in enumerate(graphs):
        corr_graph = check(graph, weakly_connected=weakly_connected)
        if corr_graph is not None:
            graphs[idx] = corr_graph
        else:
            pop.append(idx)
    return graphs, pop


def _remove_graphs(pops: list, graphs: list, idx: list) -> list:
    """Remove graphs from list.

    Args:
        pops (list): list of indices to remove.
        graphs (list): list of networkx graphs.
        idx (list): list of indices.

    Returns:
        list: remaining graphs, labels, and indices.
    """
    idx = [i for idx, i in enumerate(idx) if idx not in pops]
    graphs = [g for idx, g in enumerate(graphs) if idx not in pops]

    # relabel nodes to integers required for external embeddings
    graphs = [nx.convert_node_labels_to_integers(g)
              for g in graphs]
    graphs, idx = _check_array(graphs, idx)
    return graphs, idx


class NetworkEmbedding(BaseEstimator, TransformerMixin):
    """Creates Embedding from graph data"""

    def __init__(self,
                 embedding,
                 remove_empty: bool = False,
                 remove_isolated: bool = True,
                 path: str = None):
        """Initializes embedding

        Args:
            embedding (_type__): embedding class.
            remove_empty (bool, optional): remove empty graphs.
                Defaults to False.
            remove_isolated (bool, optional): removes isolated nodes.
                This is only relevant for connected preprocessing.
                Defaults to False.
            path (str, optional): path to save embedding.
                Defaults to None.
        """
        self.embedding = embedding
        self.remove_empty = remove_empty
        self.remove_isolated = remove_isolated
        self.path = path
        self._excluded_graphs = None
        self._isolated_idx = []

    def _set_obj_params(self, **parameters):
        """Sets object parameters. """

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _get_preprocessing(self):
        """Gets preprocessing method"""

        type_preprocessing = {
            'Embedding': self._embedding_preprocessing,
            'Connectivity': self._connectivity_preprocessing,
            'otherwise': self._connected_preprocessing,
        }
        function_preprocessing = type_preprocessing.get(
            self.embedding.__class__.__name__,
            self._connected_preprocessing
        )
        return function_preprocessing

    def _embedding_preprocessing(self):
        """Preprocessing for embedding class"""

        if not (all(isinstance(d, dict) for d in self.data)):
            graphs = self.data
            raise ValueError('Data must be a list of dicts')

        graphs = self.data
        graphs, pops = _check_consitency(graphs, False)

        if len(pops) > 0 and self.remove_empty:
            self._excluded_graphs = pops
            graphs, self.idx = _remove_graphs(pops, graphs, self.idx)
            warning(f'{len(pops)} graphs were removed from the dataset')

        self._data = graphs

    def _connectivity_preprocessing(self):
        """Preprocessing for connecticity vector"""

        if (all(isinstance(d, nx.Graph) for d in self.data)):
            graphs = self.data

        elif (hasattr(self, 'path') and
              self.path is not None):
            path = Path(self.path)
            graphs = [nx.read_gexf(path / f)
                      for f in self.data]
        else:
            graphs = [nx.read_gexf(f)
                      for f in self.data]

        graphs, pops = _check_consitency(graphs, False)
        if len(pops) > 0 and self.remove_empty:
            self._excluded_graphs = pops
            graphs, self.idx = _remove_graphs(pops, graphs, self.idx)
            warning(f'{len(pops)} graphs were removed from the dataset')
        self._data = graphs

    def _connected_preprocessing(self):
        """Preprocessing for weakly connected embeddings"""

        if (all(isinstance(d, nx.Graph) for d in self.data)):
            graphs = self.data
        elif (hasattr(self, 'path') and
              self.path is not None):
            path = Path(self.path)
            graphs = [nx.read_gexf(path / f)
                      for f in self.data]
        else:
            graphs = [nx.read_gexf(f)
                      for f in self.data]

        # remove isolated nodes
        if self.remove_isolated:
            graphs, isolated_idx = _check_isolated_nodes(graphs)
            self._isolated_idx = isolated_idx

        # package requires integers as node labels
        graphs = [nx.convert_node_labels_to_integers(g)
                  for g in graphs]

        graphs, pops = _check_consitency(graphs, True)
        if len(pops) > 0:
            self._excluded_graphs = pops
            graphs, self.idx = _remove_graphs(pops, graphs, self.idx)
            warning(f'{len(pops)} graphs were removed from the dataset')
        self._data = graphs

    @staticmethod
    def encoder(input_labels: list, input_confounds: list = None) -> tuple:
        """Encodes labels.

        Args:
            input_labels (list): iterable labels.
            input_confounds (list, optional): iterable confounds.

        Returns:
            tuple: encoded labels, confounds, and classes.
        """

        # labels
        le_labels = preprocessing.LabelEncoder()
        le_labels.fit(input_labels)
        classes = list(le_labels.classes_)
        labels = le_labels.transform(input_labels)

        # confounds
        encoded_confounds = None
        if input_confounds is not None:
            le_confounds = preprocessing.LabelEncoder()
            le_confounds.fit(input_confounds)
            encoded_confounds = le_confounds.transform(input_confounds)

            # reshape confounds
            encoded_confounds = encoded_confounds.reshape(-1, 1)

        return labels, encoded_confounds, classes

    def plot_lookup(self, feature: int, threshold: float, mni: pd.DataFrame):
        """Plots lookup / factorization of embedding.

        Args:
            feature (int): id of feature to plot.
            threshold (float): threshold for features with
                relevant contributions.
            mni (pd.DataFrame): MNI coordinates.

        Returns:
            list: glas brain plots.
        """

        if (not hasattr(self, 'embedding') or
           not hasattr(self.embedding, '_nodes')):
            pass

        relevant_mni = mni.loc[self.embedding._nodes, :]
        fact = self.embedding.get_factorization()

        scaled_fact = (
            MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(fact)
        )
        thr = np.quantile(np.concatenate(scaled_fact).ravel(), threshold)
        scaled_fact[scaled_fact < thr] = 0
        relevant_mni['contributes'] = scaled_fact[feature] > 0

        cmap = matplotlib.cm.get_cmap('GnBu')
        relevant_mni['colors'] = [cmap(v) for v in scaled_fact[feature]]
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        fig, axs = plt.subplots(nrows=1, figsize=(6.4, 0.35))
        axs.imshow(gradient, aspect='auto', cmap=cmap)
        axs.set_axis_off()

        set1 = relevant_mni.loc[(relevant_mni.set == 1) &
                                (relevant_mni.contributes),
                                ['X', 'Y', 'Z']]
        set2 = relevant_mni.loc[(relevant_mni.set == 2) &
                                (relevant_mni.contributes),
                                ['X', 'Y', 'Z']]

        colors_set1 = relevant_mni.loc[(relevant_mni.set == 1) &
                                       (relevant_mni.contributes),
                                       'colors']
        colors_set2 = relevant_mni.loc[(relevant_mni.set == 2) &
                                       (relevant_mni.contributes),
                                       'colors']

        set1_plt = view_markers(set1,
                                marker_color=colors_set1,
                                marker_size=10,
                                marker_labels=list(set1.index.values))
        set2_plt = view_markers(set2,
                                marker_color=colors_set2,
                                marker_size=10,
                                marker_labels=list(set2.index.values))
        return set1_plt, set2_plt

    def check_consistency(self,
                          data: list,
                          idx: list = None) -> np.ndarray:
        """Returns inconsistent graphs.

        Args:
            data (list): iterable of graphs.
            idx (list, optional): iterable index.
                Defaults to None.

        Returns:
            np.ndarray: idx of removed.
        """

        data, idx = _check_array(data, idx)

        self.data = data
        self.idx = idx

        processing = self._get_preprocessing()
        processing()

        return self._excluded_graphs

    def fit(self, X, y=None):
        return self

    def transform(self,
                  data: list,
                  idx: list = None,
                  **kwargs) -> np.ndarray:
        """transforms data.

        Args:
            data (list): iterable of graphs.
            idx (list, optional): iterable index.
                Defaults to None.
        Returns:
            np.ndarray: embedding.
        """

        if idx is None:
            idx = np.arange(len(data))
        data, idx = _check_array(data, idx)

        self.data = data
        self.idx = idx
        self._set_obj_params(**kwargs)

        processing = self._get_preprocessing()
        processing()

        self.embedding.fit(self._data)

        self.values = self.embedding.get_embedding()
        return self.values
