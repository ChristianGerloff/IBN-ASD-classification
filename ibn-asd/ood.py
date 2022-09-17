"""Out-of-distribution evaluation."""
import os
import sys
import logging
import argparse
import mlflow
import numpy as np
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import config.config as conf

from pathlib import Path
from dataclasses import asdict
from mlflow.tracking import MlflowClient
from config.config import ClassificationParams

from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score

from interbrainnetworks.embeddings import NetworkEmbedding


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
             model_source: str,
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

    tags = {'embedding': embedding,
            'estimator': estimator,
            'stage': 'ood'}

    local_path = Path(local_path)
    embeddings_conf = load_embedding_config(embedding)
    clf_conf = ClassificationParams()

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

        network_embeddings = NetworkEmbedding(embeddings_conf.embedding)
        remove_idx = network_embeddings.check_consistency(data, data.index)
        if remove_idx is not None:
            data = data.drop(remove_idx)
            labels = labels.drop(remove_idx)

        # load model
        model = mlflow.sklearn.load_model(model_source)

    except Exception as e:
        logger.error(f'Unable to read artifacts: {e}')
        return None

    with mlflow.start_run() as active_run:
        mlflow.set_tags(tags)

        # log params
        mlflow.log_params(asdict(clf_conf))
        mlflow.log_params(asdict(embeddings_conf))
        mlflow.log_params({'classes': labels.value_counts()})

        pred = model.predict(data)
        mlflow.sklearn.eval_and_log_metrics(model,
                                            data,
                                            labels,
                                            prefix='test_',
                                            pos_label=1)
        mlflow.log_metrics({
            'test_roc_auc': roc_auc_score(labels, pred),
            'test_accuracy': accuracy_score(labels, pred),
            'test_balanced_accuracy': balanced_accuracy_score(labels, pred),
            'test_precision': precision_score(labels, pred),
            'test_recall': recall_score(labels, pred)
        })

        logger.info(f'OOD evaluation run_id {active_run.info.run_id} finished')


if __name__ == '__main__':
    """evaluation."""
    parser = argparse.ArgumentParser()
    PATH = str(Path.cwd() / str('tmp/artifact_download'))
    parser.add_argument('--embedding', type=str, default='Connectivity')
    parser.add_argument('--model_source',
                        type=str,
                        default=str(Path.cwd() / str('tmp/trained_model')))
    parser.add_argument('--graphs_metadata_filename', type=str, default='graphs_metadata.pkl')
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--artifact_path', type=str)
    parser.add_argument('--estimator', type=str, default='WCO')
    parser.add_argument('--local_path', type=str, default=PATH)

    args = parser.parse_args()

    if os.getenv('MLFLOW_TRACKING_URI', None):
        logger.info('MLFLOW_TRACKING_URI not set')

    if (args.embedding is None) or (args.model_source is None):
        raise Exception(
            'argument --embedding and --model_source should be specified'
        )

    if args.run_id and args.artifact_path:
        local_path = fetch_artifacts(args.run_id, args.artifact_path)
        evaluate(args.embedding,
                 args.model_source,
                 graphs_metadata_filename=args.graphs_metadata_filename,
                 estimator=args.estimator,
                 local_path=local_path)
    else:
        evaluate(args.embedding,
                 args.model_source,
                 estimator=args.estimator,
                 local_path=args.local_path)
