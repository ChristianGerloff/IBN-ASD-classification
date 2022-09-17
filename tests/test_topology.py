"""
Tests of topology functions.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from interbrainnetworks.networks import construct_bipartite_graphs
from interbrainnetworks.networks import calculate_topologies
from interbrainnetworks.networks.topology import _check_channels

ESTIMATOR = 'Block_1_salient_wco_scaled'


@pytest.fixture
def network_data():
    filename = 'network_data.parquet'
    network_data = pd.read_parquet(
        Path(__file__).resolve().parent / 'data' / filename
    )
    return network_data


@pytest.mark.parametrize('signal_pair,estimator',
                         [(1, ESTIMATOR),
                          pytest.param(1, 'WCO', marks=pytest.mark.xfail)])
def test_init_network(signal_pair, estimator, network_data):
    """Tests the basic calculation."""
    test_graph = network_data.loc[network_data.signal_pair < signal_pair, :]
    single_graph = construct_bipartite_graphs(test_graph, 0, estimator)

    ibn = pd.DataFrame(
        calculate_topologies(network_data, estimator).compute()
    )

    # equal results for single graph and all graphs
    subset = ibn.loc[ibn.graph_id.isin(single_graph.graph_id)]
    assert np.array_equal(single_graph.metric, subset.metric)


@pytest.mark.parametrize(
    (
        'channel_cols,'
        'coord_cols'
    ),
    [
        (
            ['Channel_1', 'Channel_2'],
            [['Channel_1_X', 'Channel_1_Z'],
             ['Channel_2_X', 'Channel_2_Z']]
        ),
        pytest.param(
            ['Channel_1', 'Channel_2'],
            [['Channel_1_X'],
             ['Channel_2_X', 'Channel_2']],
            marks=pytest.mark.xfail
        )
    ]
)
def test_check_channels(channel_cols, coord_cols, network_data):
    """Tests the coordinates check."""
    nodes, coordinates = _check_channels(network_data,
                                         channel_cols,
                                         coord_cols)

    assert nodes is not None
    assert coordinates is not None
