"""
Tests for embedding class.
"""
import pytest
import pandas as pd
from pathlib import Path
from interbrainnetworks.embeddings import Embedding

PATH = Path(__file__).resolve().parent / 'data' / 'graphs'


@pytest.fixture
def ibn():
    filename = 'ibn.pkl'
    network_data = pd.read_pickle(
        Path(__file__).resolve().parent / 'data' / filename
    )
    return network_data


@pytest.fixture
def params():
    params = {
        'n_components': 10,
        'init': 'nndsvd',
        'solver': 'cd',
        'tol': 1e-4,
        'max_iter': 10000,
        'beta_loss': 'frobenius',
        'seed': 20211001
    }
    return params


def test_init_embedding(ibn, params):
    graphs = list(ibn['metric'].values)
    embedding = Embedding(**params)
    embedding.fit(graphs)
    coefficients = embedding.get_embedding()

    assert coefficients.shape == (len(graphs), params['n_components'])
