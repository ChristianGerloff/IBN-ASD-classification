"""
Tests of synthetic toy data.
"""

import pytest
import numpy as np
from interbrainnetworks.networks.testdata import _create_sample


@pytest.mark.parametrize(
    ('idx,conditions,'
     'partners,'
     'channels,'
     'all_idx,'
     'badchannels'),
    [(1,
      ['cooperation'],
      ['peer', 'stranger'],
      np.arange(1, 23),
      np.arange(1, 5),
      True),
     (2,
      ['cooperation'],
      ['peer', 'stranger'],
      np.arange(1, 23),
      np.arange(1, 6),
      False)])
def test_create_sample(idx,
                       conditions,
                       partners,
                       channels,
                       all_idx,
                       badchannels):
    """Tests the exclusion of bad channels."""
    (task, rest,
     task_permu, rest_permu) = _create_sample(idx=idx,
                                              conditions=conditions,
                                              partners=partners,
                                              channels=channels,
                                              all_idx=all_idx,
                                              badchannels=badchannels)

    n_actual = len(conditions) * len(partners) *len(channels) **2
    n_permu = n_actual * (len(all_idx) - 1)
    if badchannels is False:
        assert len(task) == n_actual
        assert len(rest) == n_actual
        assert len(task_permu) == n_permu
        assert len(rest_permu) == n_permu
    else:
        n_lower_actual = n_actual * 0.8
        n_lower_permu = n_lower_actual * (len(all_idx) - 1)
        assert (len(task) >= n_lower_actual and
                len(task) <= n_actual)
        assert (len(rest) >= n_lower_actual and
                len(rest) <= n_actual)
        assert (len(task_permu) >= n_lower_permu and
                len(task_permu) <= n_permu)
        assert (len(rest_permu) >= n_lower_permu and
                len(rest_permu) <= n_permu)
