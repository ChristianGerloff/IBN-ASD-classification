"""
Tests of networks module.
"""
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from interbrainnetworks.networks import Networks


@pytest.fixture
def mni_data():
    filename = 'channel2mni.csv'
    mni = pd.read_csv(
            Path(__file__).resolve().parent / 'data' / filename,
            dtype={'Channel': np.int32, 'X': np.float64, 'Y': np.float64,
                   'Z': np.float64, 'Area': 'str'},
            header=0,
            index_col=0
        )
    return mni


@pytest.fixture
def inputdata():

    path = Path(__file__).resolve().parent / 'data'
    files = ['task.parquet',
             'rest.parquet',
             'task_permutation.parquet',
             'rest_permutation.parquet']
    data = [pd.read_parquet(path / f) for f in files]
    return data


@pytest.fixture
def init_parameters():
    params = {
        'dyad_type': 'actual',
        'chromophore': 'Hbo',
        'atlas': 'default',
        'channel_set': 22,
        'input_connectivity_estimator': 'salient_wco',
        'condition_filter': [],
        'partner_filter': [],
        'id_filter': [1],
    }
    return params


@pytest.fixture
def set_parameters(init_parameters):
    params = init_parameters
    params.update(
        {
            # name of task and restblocks
            'blocks': ['Block_1', 'Block_2', 'Block_0'],
            'factors_scaling': ['channel_pair'],
            'connectivity_scaling': True,
            'exchangeables': ['channel_pair', 'Condition', 'Partner'],
            'alpha': 0.05,
        }
    )
    return params


def test_init_network(init_parameters):
    """Tests initialization of Networks object."""
    ib_network = Networks(**init_parameters)
    assert ib_network.dyad_type == 'actual'
    assert ib_network.condition_filter == init_parameters['condition_filter']
    assert ib_network.partner_filter == init_parameters['partner_filter']


@pytest.mark.parametrize(
    (
        'dyad_type,'
        'chromophore,'
        'atlas,'
        'condition_filter,'
        'partner_filter,'
        'id_filter'
    ),
    [
        ('actual', 'Hbo', None, None, None, None),
        ('actual', 'Hbo', 'default', ['cooperation'], ['peer'], [1])
    ]
)
def test_set_data(dyad_type,
                  chromophore,
                  atlas,
                  condition_filter,
                  partner_filter,
                  id_filter,
                  inputdata,
                  mni_data):
    """Tests initial data wrangling."""
    dataset = Networks(dyad_type,
                       chromophore=chromophore,
                       atlas=atlas,
                       condition_filter=condition_filter,
                       partner_filter=partner_filter,
                       id_filter=id_filter)
    dataset.set_data(inputdata[0], inputdata[1], mni_data)

    # test chromophore filter
    assert dataset.data.chromophore.unique() == [chromophore]

    # test condition filter
    if condition_filter is not None:
        assert dataset.data.Condition.unique() == condition_filter

    # test partner filter
    if partner_filter is not None:
        assert dataset.data.Partner.unique() == partner_filter

    # test id filter
    if id_filter is not None and dyad_type == 'actual':
        assert ~any(np.isin(dataset.data.ID.values, id_filter))


@pytest.mark.parametrize(
    (
        'separated_estimators,'
        'expected_perc'
    ),
    [
        (True, 7.9),
        (False, 7.9),
        pytest.param(True, 90, marks=pytest.mark.xfail),
        pytest.param(False, 7.9, marks=pytest.mark.xfail)
    ]
)
def test_null_dist(separated_estimators,
                   expected_perc,
                   inputdata,
                   mni_data,
                   set_parameters):
    """Test null distribution scaling."""
    params = set_parameters
    actual_set = Networks(**params,
                          separated_estimators=separated_estimators)
    actual_set.set_data(inputdata[0], inputdata[1], mni_data)

    scaling_factors = actual_set.scaling_factor
    if separated_estimators is True:
        assert scaling_factors.shape[1] == (
            len(params['factors_scaling'])+len(params['blocks']))
    else:
        assert scaling_factors.shape[1] == (
            len(params['factors_scaling'])+1)

    # ensure that the percentage of excluded channels is plausible
    perc = actual_set.excluded_channels
    assert np.allclose(perc, expected_perc, rtol=.2)

    # test name of block estimators
    name_estimators = actual_set.block_estimators_scaled
    for i in name_estimators:
        assert any(s in i for s in params['blocks'])

    # test scaling of block estimators
    if params['connectivity_scaling']:
        actual_set.scale()
    actual_set.transform(actual_set.data)

    if separated_estimators is True:
        assert actual_set._null_dist_threshold.shape == (
            len(actual_set.data.groupby(params['exchangeables'])),
            len(params['exchangeables'])+len(name_estimators))
    else:
        assert actual_set._null_dist_threshold.shape == (
            len(actual_set.data.groupby(params['exchangeables'])),
            len(params['exchangeables'])+1)
