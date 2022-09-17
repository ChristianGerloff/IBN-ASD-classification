"""Search space configurations"""
from skopt.space import Real, Categorical
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC

from karateclub import LDP, WaveletCharacteristic
from karateclub import FeatherGraph, GL2Vec, Graph2Vec, GeoScattering
from interbrainnetworks.embeddings import Connectivity, Embedding

from interbrainnetworks.embeddings import NetworkEmbedding


class SearchSpace(object):
    def __init__(self, embedding, classifier, seed: int = 77617665):
        self.embedding = embedding
        self.classifier = classifier
        self.seed = seed
        pass

    @property
    def space(self):
        cls_dict = {
            'RidgeClassifier': self.ridge_sp(),
            'LinearSVC': self.linsvc_sp(),
        }
        emb_dict = {
            'FeatherGraph': self.feather_sp(),
            'GL2Vec': self.gl2vec_sp(),
            'Graph2Vec': self.graph2Vec_sp(),
            'GeoScattering': self.geoscattering_sp(),
            'WaveletCharacteristic': self.diffusionwavelet_sp(),
            'Embedding': self.ibn_sp(),
            'Connectivity': self.connectivity_sp(),
            'LDP': self.ldp_sp(),
        }
        space_dict = cls_dict.get(self.classifier, self.ridge_sp())
        space_dict.update(emb_dict.get(self.embedding, self.feather_sp()))
        return space_dict

    def ridge_sp(self) -> dict:
        param_sp = {
            'scaler': Categorical([None, StandardScaler()]),
            'classifier': Categorical([RidgeClassifier()]),
            'classifier__alpha': Categorical([0.2, 0.4, 0.6, 0.8]),
            'classifier__tol': Real(1e-8, 1e-4, prior='log-uniform'),
            'classifier__class_weight': Categorical(['balanced']),
            'classifier__normalize': Categorical([True]),
            'classifier__random_state': Categorical([self.seed]),
        }
        return param_sp

    def linsvc_sp(self) -> dict:
        param_sp = {
            'scaler': Categorical([None, StandardScaler()]),
            'classifier': Categorical([LinearSVC()]),
            'classifier__dual': Categorical([False]),
            'classifier__penalty': Categorical(['l1', 'l2']),
            'classifier__loss': Categorical(['squared_hinge']),
            'classifier__fit_intercept': Categorical([True, False]),
            'classifier__C': Real(1e-6, 1e+6, prior='log-uniform'),
            'classifier__class_weight': Categorical(['balanced']),
            'classifier__random_state': Categorical([self.seed]),
        }
        return param_sp

    def connectivity_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical([NetworkEmbedding(Connectivity())]),
        }
        return param_sp

    def ibn_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(Embedding(n_components=4, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=5, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=8, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=10, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=12, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=15, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=20, log=True, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=4, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=5, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=8, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=10, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=12, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=15, log=False, seed=self.seed)),
                 NetworkEmbedding(Embedding(n_components=20, log=False, seed=self.seed))]
            ),
        }
        return param_sp

    def ldp_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(LDP(bins=2), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=4), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=6), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=8), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=10), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=16), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=24), remove_isolated=True),
                 NetworkEmbedding(LDP(bins=32), remove_isolated=True)]
            ),
        }
        return param_sp

    def feather_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(FeatherGraph(order=2, seed=self.seed)),
                 NetworkEmbedding(FeatherGraph(order=5, seed=self.seed)),
                 NetworkEmbedding(FeatherGraph(order=2, pooling='min', seed=self.seed)),
                 NetworkEmbedding(FeatherGraph(order=5, pooling='min', seed=self.seed)),
                 NetworkEmbedding(FeatherGraph(order=2, pooling='max', seed=self.seed)),
                 NetworkEmbedding(FeatherGraph(order=5, pooling='max', seed=self.seed))]),
        }
        return param_sp

    def gl2vec_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(GL2Vec(wl_iterations=2, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=5, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=10, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=2, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=5, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=10, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=2,
                                         min_count=2,
                                         learning_rate=0.01,
                                         seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=5,
                                         min_count=2,
                                         learning_rate=0.01,
                                         seed=self.seed)),
                 NetworkEmbedding(GL2Vec(wl_iterations=10,
                                         min_count=2,
                                         learning_rate=0.01,
                                         seed=self.seed))]),
        }
        return param_sp

    def graph2Vec_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(Graph2Vec(wl_iterations=2, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=5, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=10, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=2, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=5, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=10, learning_rate=0.01, seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=2,
                                            min_count=2,
                                            learning_rate=0.01,
                                            seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=5,
                                            min_count=2,
                                            learning_rate=0.01,
                                            seed=self.seed)),
                 NetworkEmbedding(Graph2Vec(wl_iterations=10,
                                            min_count=2,
                                            learning_rate=0.01,
                                            seed=self.seed))]),
        }
        return param_sp

    def geoscattering_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical(
                [NetworkEmbedding(GeoScattering(order=2, moments=4, seed=self.seed)),
                 NetworkEmbedding(GeoScattering(order=2, moments=2, seed=self.seed)),
                 NetworkEmbedding(GeoScattering(order=4, moments=2, seed=self.seed)),
                 NetworkEmbedding(GeoScattering(order=4, moments=4, seed=self.seed))]),
        }
        return param_sp

    def diffusionwavelet_sp(self) -> dict:
        param_sp = {
            'embedding': Categorical([
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=25, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=25, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=5, eval_points=25, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=20, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=20, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=5, eval_points=20, theta_max=2.5, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=25, theta_max=2, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=25, theta_max=2, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=20, theta_max=2, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=20, theta_max=2, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=5, eval_points=20, theta_max=2, pooling='mean')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=25, theta_max=2, pooling='max')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=25, theta_max=2, pooling='max')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=5, eval_points=25, theta_max=2, pooling='max')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=3, eval_points=20, theta_max=2.5, pooling='max')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=4, eval_points=20, theta_max=2.5, pooling='max')
                ),
                NetworkEmbedding(
                    WaveletCharacteristic(order=5, eval_points=20, theta_max=2.5, pooling='max')
                )]
            ),
        }
        return param_sp
