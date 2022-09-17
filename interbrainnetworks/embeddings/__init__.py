"""
The :mod:`networks` module to generate
interbrain network embedding.
"""

from .networkembedding import NetworkEmbedding
from .connectivity import Connectivity
from .embedding import Embedding

__all__ = [
    'NetworkEmbedding',
    'Embedding',
    'Connectivity'
]
