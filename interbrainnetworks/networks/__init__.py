"""
The :mod:`networks` module to generate
interbrain networks.
"""

from .networks import Networks
from .topology import construct_bipartite_graphs, calculate_topologies
from .testdata import create_testdata

__all__ = [
    'Networks',
    'construct_bipartite_graphs',
    'calculate_topologies',
    'create_testdata'
]
