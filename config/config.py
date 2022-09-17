"""General configurations"""
from dataclasses import dataclass, field
from typing import List

from karateclub import LDP, WaveletCharacteristic
from karateclub import FeatherGraph, GL2Vec, Graph2Vec, GeoScattering
from interbrainnetworks.embeddings import Connectivity, Embedding


@dataclass
class NetworkParams:
    """Network parameters."""
    chromophore: str = 'Hbo'
    atlas: str = 'default'
    channel_set: int = 22
    condition_filter: List[str] = field(default_factory=list)
    partner_filter: List[str] = field(default_factory=list)
    id_filter: List[str] = field(default_factory=list)
    group_filter: List[int] = field(default_factory=lambda: [3])
    control_group: int = 2
    asd_group: int = 1
    separated_estimators: bool = True
    blocks: List[str] = field(default_factory=lambda: ['Block_2', 'Block_4'])
    input_connectivity_estimator: str = 'WCO_shannon'
    connectivity_scaling: bool = True
    factors_scaling: List[str] = field(
        default_factory=lambda: ['channel_pair', 'Condition', 'Partner'])
    exchangeables: List[str] = field(
        default_factory=lambda: ['channel_pair', 'Condition', 'Partner'])
    alpha: float = 0.2
    parent_run_id: str = None


@dataclass
class ClassificationParams:
    """General classification settings."""
    dim_opt: int = 20
    iter_opt: int = 20
    points_opt: int = 1
    inner_n_splits: int = 3
    outer_n_splits: int = 5
    outer_n_repeats: int = 10
    seed: int = 20211001
    scoring: str = 'roc_auc'
    groups: str = 'ID'
    cv_scores: List[str] = field(
        default_factory=lambda: [
            'roc_auc',
            'accuracy',
            'balanced_accuracy',
            'precision',
            'recall'
        ]
    )


@dataclass
class ConnectivityEmbeddingParams:
    """Connectivity embedding parameters."""
    embedding_name: str = 'Connectivity'
    field: str = 'filename'
    graph_type: str = 'interpolated_graph'
    remove_isolated: bool = False
    embedding = Connectivity()


@dataclass
class IbnEmbeddingParams:
    """Interbrain network embedding parameters."""
    embedding_name: str = 'Embedding'
    field: str = 'density'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = False
    embedding = Embedding()


@dataclass
class LDPParams:
    """LDP embedding parameters."""
    embedding_name: str = 'LDP'
    field: str = 'filename'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = True
    embedding = LDP(bins=5)


@dataclass
class FeatherGraphParams:
    """Feather embedding parameters."""
    embedding_name: str = 'FeatherGraph'
    field: str = 'filename'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = True
    embedding = FeatherGraph()


@dataclass
class GL2VecParams:
    """GL2Vec embedding parameters."""
    embedding_name: str = 'GL2Vec'
    field: str = 'filename'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = True
    embedding = GL2Vec()


@dataclass
class Graph2VecParams:
    """Graph2Vec embedding parameters."""
    embedding_name: str = 'Graph2Vec'
    field: str = 'filename'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = True
    embedding = Graph2Vec()


@dataclass
class GeoScatteringParams:
    """GeoScattering embedding parameters."""
    embedding_name: str = 'GeoScattering'
    field: str = 'filename'
    graph_type: str = 'interpolated_graph'
    remove_isolated: bool = False
    embedding = GeoScattering()


@dataclass
class WaveletCharacteristicParams:
    """WaveletCharacteristic embedding parameters."""
    embedding_name: str = 'WaveletCharacteristic'
    field: str = 'filename'
    graph_type: str = 'interbrainnetwork'
    remove_isolated: bool = True
    embedding = WaveletCharacteristic()
