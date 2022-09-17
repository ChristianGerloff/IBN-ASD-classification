"""Nested stratified cross-validation."""
import os
import sys
import logging
import argparse
import mlflow
import numpy as np
import pandas as pd

from pathlib import Path
from dataclasses import asdict
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score
from skopt import BayesSearchCV

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import config.config as conf
from config.config import ClassificationParams
from config.search import SearchSpace
from interbrainnetworks.embeddings import NetworkEmbedding


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ToDo nach utils auslagern
def fetch_artifacts(run_id: str,
                    artifact_path: str,
                    local_path: str = '/tmp/artifact_download') -> Path:
    """Fetch artifacts from MLflow run.

    Args:
        run_id (str): MLflow run ID.
        artifact_path (str): MLflow artifact path.
        local_path (str, optional): local path to save artifacts.
            Defaults to '/tmp/artifact_download'.

    Returns:
        Path: local path to downloaded artifacts
    """
    local_path = Path(local_path) / run_id
    if local_path.is_dir():
        return local_path / artifact_path

    local_path.mkdir(parents=True, exist_ok=True)

    try:
        client = MlflowClient()
        client.download_artifacts(run_id, artifact_path, local_path)
    except Exception as e:
        logger.error(f'Unable to download artifact of run {run_id}: {e}')
    return local_path / artifact_path


def mlflow_log_metric_array(metric: str, data: np.ndarray):
    """Log metric array to MLflow.


    Args:
        metric (str): metric name
        data (np.ndarray): metric array
    """
    for idx, value in enumerate(data):
        mlflow.log_metric(key=metric, value=value, step=idx)


def mlflow_parse_skopt_params(opt_params: dict, prefix: str = 'best_') -> dict:
    """Parse skopt parameters to MLflow format.

    Args:
        opt_params (dict): skopt parameters
        prefix (str, optional): prefix for parameter names.
            Defaults to 'best_'.

    Returns:
        dict: parameters
    """
    params_dict = {}
    for key in opt_params:
        item = opt_params[key]

        # check if type is valid for mlflow params
        direct_types = ['int', 'float', 'dict', 'str']
        directs = [isinstance(item, eval(i)) for i in direct_types]
        if any(directs):
            params_dict.update({
                prefix + key: opt_params[key]
            })
        else:
            params_dict.update({
                prefix + key: opt_params[key].__class__.__name__
            })
    return params_dict


def load_embedding_config(embedding: str):
    """Load embedding configuration.

    Args:
        embedding (str): embedding name

    Returns:
        dict: embedding configuration
    """
    params = {
        'Connectivity': conf.ConnectivityEmbeddingParams(),
        'Embedding': conf.IbnEmbeddingParams(),
        'LDP': conf.LDPParams(),
        'FeatherGraph': conf.FeatherGraphParams(),
        'GL2Vec': conf.GL2VecParams(),
        'Graph2Vec': conf.Graph2VecParams(),
        'GeoScattering': conf.GeoScatteringParams(),
        'WaveletCharacteristic': conf.WaveletCharacteristicParams()
    }
    return params.get(embedding,
                      conf.ConnectivityEmbeddingParams())


def evaluate(embedding: str,
             classifier: str,
             graphs_metadata_filename: str = 'graphs_metadata.pkl',
             estimator: str = 'WCO',
             local_path: str = '/tmp/artifact_download'):
    """Evaluate embedding.

    Args:
        embedding (str): embedding name
        classifier (str): classifier name
        graphs_metadata_filename (str, optional): graphs metadata filename.
            Defaults to 'graphs_metadata.pkl'.
        estimator (str, optional): estimator name.
            Defaults to 'WCO'.
        local_path (str, optional): local path to save artifacts.
            Defaults to '/tmp/artifact_download'.

    Returns:
        dict: evaluation results
    """
    tags = {'stage': 'cv',
            'embedding': embedding,
            'classifier': classifier,
            'estimator': estimator}

    local_path = Path(local_path)
    embeddings_conf = load_embedding_config(embedding)
    clf_conf = ClassificationParams()

    with mlflow.start_run() as active_run:
        mlflow.set_tags(tags)
        # read files
        try:
            graphs = pd.read_pickle(local_path / graphs_metadata_filename)

            # filter graphs
            graphs = graphs.loc[graphs.graph_type == embeddings_conf.graph_type, :]

            graphs = graphs.reset_index(drop=True)
            if embeddings_conf.field == 'filename':
                data = str(local_path) + '/' + graphs[embeddings_conf.field]
            else:
                data = graphs[embeddings_conf.field]
            labels = graphs['label']
            groups = graphs[clf_conf.groups]

            network_embeddings = NetworkEmbedding(
                embeddings_conf.embedding,
                remove_isolated=embeddings_conf.remove_isolated)
            remove_idx = network_embeddings.check_consistency(
                data,
                data.index
            )
            n_isolated = len(network_embeddings._isolated_idx)

            if remove_idx is not None:
                data = data.drop(remove_idx)
                labels = labels.drop(remove_idx)
                groups = groups.drop(remove_idx)
            data[:] = network_embeddings._data.copy()

        except Exception as e:
            logger.error(f'Unable to read artifacts: {e}')
            return None

        # default pipeline
        pipe = Pipeline(
            [
                ('embedding', network_embeddings),
                ('scaler', StandardScaler()),
                ('classifier', eval(classifier)())
            ]
        )

        # initialize search space
        search_space = SearchSpace(embedding,
                                   classifier,
                                   clf_conf.seed)

        # load search space
        inner_cv = StratifiedGroupKFold(n_splits=clf_conf.inner_n_splits,
                                        shuffle=True,
                                        random_state=clf_conf.seed)
        outer_cv = StratifiedGroupKFold(n_splits=clf_conf.outer_n_splits,
                                        shuffle=True,
                                        random_state=clf_conf.seed)

        model = BayesSearchCV(
            pipe,
            [(search_space.space, clf_conf.dim_opt)],
            cv=inner_cv,
            n_iter=clf_conf.iter_opt,
            n_jobs=-1,
            n_points=clf_conf.points_opt,
            return_train_score=True,
            scoring=clf_conf.scoring
        )

        roc = []
        acc = []
        balanced_acc = []
        precision = []
        recall = []
        n_class_train = []
        n_class_test = []

        i = 0
        for train_idx, test_idx in outer_cv.split(data, labels, groups):
            i += 1
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            labels_train, labels_test = labels.iloc[train_idx], labels.iloc[test_idx]
            subgroups = groups.iloc[train_idx]

            # shuffle
            labels_train = labels_train.sample(
                frac=1, random_state=clf_conf.seed * i).reset_index(drop=True)

            _, counts_train = np.unique(labels_train, return_counts=True)
            _, counts_test = np.unique(labels_test, return_counts=True)
            n_class_train.append(counts_train[1])
            n_class_test.append(counts_test[1])

            if 0 in counts_test or 0 in counts_train:
                logger.warning('Fold {i} contains not all classes and was skipped')
                continue

            model.fit(train, labels_train, groups=subgroups)
            pred = model.predict(test)

            # performance metrics
            roc.append(roc_auc_score(labels_test, pred))
            acc.append(accuracy_score(labels_test, pred))
            balanced_acc.append(balanced_accuracy_score(labels_test, pred))
            precision.append(precision_score(labels_test, pred))
            recall.append(recall_score(labels_test, pred))

        metrics = {
            'roc_auc': roc,
            'accuracy': acc,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
        }

        # random shuffle again
        labels = labels.sample(frac=1, random_state=clf_conf.seed).reset_index(drop=True)
        model.fit(data, labels, groups=groups)

        # log model
        mlflow.sklearn.log_model(model, 'trained_model')
        # mlflow.sklearn.log_model(pipe, "model", signature=signature)

        # log params
        mlflow.log_params(asdict(clf_conf))
        mlflow.log_params(asdict(embeddings_conf))
        mlflow.log_params(mlflow_parse_skopt_params(model.best_params_, 'ood_sample_'))
        mlflow.log_params({'classes': labels.value_counts(), 'n_isolated': n_isolated})

        # log cv metrics
        for key, values in metrics.items():
            mlflow.log_metrics({'test_' + key: np.nanmean(values)})
            mlflow_log_metric_array('details_' + 'test_' + key, values)
        mlflow_log_metric_array('train_n_asd', n_class_train)
        mlflow_log_metric_array('test_n_asd', n_class_test)
        logger.info(f'Within sample evaluation run_id {active_run.info.run_id} finished')


if __name__ == '__main__':
    """evaluation."""
    parser = argparse.ArgumentParser()

    PATH = str(Path.cwd() / str('tmp/artifact_download'))
    parser.add_argument('--embedding', type=str, default='LDP')
    parser.add_argument('--classifier', type=str, default='LinearSVC')
    parser.add_argument('--graphs_metadata_filename', type=str, default='graphs_metadata.pkl')
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--artifact_path', type=str)
    parser.add_argument('--estimator', type=str, default='WCO')
    parser.add_argument('--local_path', type=str, default=PATH)

    args = parser.parse_args()

    if os.getenv('MLFLOW_TRACKING_URI', None):
        logger.info('MLFLOW_TRACKING_URI not set')

    if (args.embedding is None) or (args.classifier is None):
        raise Exception(
            'argument --embedding and --classifier should be specified'
        )

    if args.run_id and args.artifact_path:
        local_path = fetch_artifacts(args.run_id, args.artifact_path)
        evaluate(args.embedding,
                 args.classifier,
                 graphs_metadata_filename=args.graphs_metadata_filename,
                 estimator=args.estimator,
                 local_path=local_path
                 )
    else:
        evaluate(args.embedding,
                 args.classifier,
                 estimator=args.estimator,
                 local_path=args.local_path)
