"""
Tests for network embedding class.
"""

import pytest
import pandas as pd
from pathlib import Path
from interbrainnetworks.embeddings import NetworkEmbedding
from interbrainnetworks.embeddings import Embedding


PATH = Path(__file__).resolve().parent / 'data' / 'graphs'


@pytest.fixture
def ibn():
    filename = 'ibn.pkl'
    network_data = pd.read_pickle(
        Path(__file__).resolve().parent / 'data' / filename
    )
    return network_data


@pytest.mark.parametrize('embedding', [Embedding()])
def test_init_networkembedding(embedding, ibn):
    # test init with default parameters
    network_embedding = NetworkEmbedding(embedding)

    assert (
        network_embedding.embedding.__class__.__name__ ==
        embedding.__class__.__name__
    )


@pytest.mark.parametrize(
    (
        'embedding,'
        'data_col,'
        'label_col,'
        'idx_col'
    ),
    [
        (Embedding(), 'metric', 'partner', 'graph_id'),
        (Embedding(), 'metric', 'condition', 'graph_id'),
    ]
)
def test_fit_networkembedding(embedding,
                              data_col,
                              label_col,
                              idx_col,
                              ibn):
    # test init with default parameters
    network_embedding = NetworkEmbedding(embedding)

    data = ibn[data_col].values
    labels = ibn[label_col].values
    idx = ibn[idx_col].values

    network_embedding.fit_transform(data, labels, idx, path=PATH)
