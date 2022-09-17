"""
Tests of connectivity class.
"""

import pytest
import pandas as pd
import networkx as nx
from pathlib import Path
from interbrainnetworks.embeddings import Connectivity

PATH = Path(__file__).resolve().parent / 'data' / 'graphs'


@pytest.fixture
def ibn():
    filename = 'ibn.pkl'
    network_data = pd.read_pickle(
        Path(__file__).resolve().parent / 'data' / filename
    )
    return network_data


def test_init_connectivity(ibn):
    graphs = [
        nx.read_gexf(PATH / f)
        for f in ibn['filename'].values
    ]

    subset = graphs[0:4]
    embedding = Connectivity()
    embedding.fit(subset)
    values = embedding.get_embedding()

    assert values.shape[0] == len(subset)
