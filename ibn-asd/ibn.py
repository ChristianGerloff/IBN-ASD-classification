"""Constructs networks from hyperscanning data."""

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

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from interbrainnetworks.networks import Networks
from interbrainnetworks.networks import calculate_topologies

from config.config import NetworkParams

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


def network_generation(actual_filename: str,
                       shuffled_filename: str,
                       network_params: dict,
                       cohort_filename: str = 'cohort.csv',
                       mni_filename: str = 'channel2mni.csv',
                       local_path: str = '/tmp/artifact_download',
                       output_path: str = '/tmp/processing'):
    """Generates interbrain networks from hyperscanning data.

    Args:
        actual_filename (str): actual data filename
        shuffled_filename (str): shuffled data filename
        network_params (NetworkParams): network parameters.
        cohort_filename (str, optional): cohort filename.
            Defaults to 'cohort.csv'.
        mni_filename (str, optional): mni filename.
            Defaults to 'channel2mni.csv'.
        local_path (str, optional): local path to data.
            Defaults to '/tmp/artifact_download'.
        output_path (str, optional): local path to save output.
            Defaults to '/tmp/processing'.
    """

    CE_NAME = 'input_connectivity_estimator'
    local_path = Path(local_path)

    # read files
    try:
        mni = None
        if mni_filename:
            mni = pd.read_csv(local_path / mni_filename,
                              dtype={'Channel': np.int32,
                                     'X': np.float64,
                                     'Y': np.float64,
                                     'Z': np.float64,
                                     'Area': 'str'},
                              header=0,
                              index_col=0)

        cohort = pd.read_csv(local_path / cohort_filename)
        actual_raw = pd.read_parquet(local_path / actual_filename)
        shuffled_raw = pd.read_parquet(local_path / shuffled_filename)

    except Exception as e:
        logger.error(f'Unable to read artifacts: {e}')
        return None

    with mlflow.start_run() as active_run:
        mlflow.set_tag('stage', 'ibn')
        mlflow.set_tag('estimator',
                       network_params[CE_NAME])
        mlflow.set_tag('chromophore', network_params['chromophore'])

        # create folder
        output_path = Path(output_path) / active_run.info.run_id
        output_path.mkdir(parents=True, exist_ok=True)

        # filter unmatched control subjects
        cohort = cohort.drop_duplicates()
        if network_params['group_filter']:
            ids = cohort.loc[cohort.Group.isin(network_params['group_filter']), 'ID']
            ids = list(ids.values)
            network_params['id_filter'].extend(ids)

        cohort = cohort.set_index(['ID'])

        # specify actual pairs
        actual_net = Networks('actual', **network_params)
        actual_net.set_data(actual_raw, None, mni)

        actual_net.data.loc[:, 'label'] = (
            cohort.loc[actual_net.id_labels[actual_net.data.ID], 'Group'].values
        )

        # specify shuffled pairs
        shuffled_net = Networks('permu', **network_params)
        shuffled_net.set_data(shuffled_raw, None, mni)
        shuffled_net.data.loc[:, 'label'] = (
            cohort.loc[shuffled_net.id_1_labels[shuffled_net.data.ID_1], 'Group'].values
        )

        shuffled_net.scale()
        mean_estimators = shuffled_net.scaling_factor
        actual_net.transform(shuffled_net.data, mean_estimators)

        actual_net.data['label'] = (
            actual_net.data['label'].replace(network_params['control_group'], 0)
        )
        actual_net.data['label'] = (
            actual_net.data['label'].replace(network_params['asd_group'], 1)
        )

        graph_results = pd.DataFrame()
        task = actual_net.data.loc[actual_net.data.Condition != 'rest', :]
        for i in actual_net.block_estimators_scaled:
            query = calculate_topologies(
                task, i, path=output_path).compute()
            graph_results = pd.concat([graph_results, query])

        # save results
        actual_net.data.to_pickle(output_path / 'actual_net.pkl')
        graph_results.to_pickle(output_path / 'graphs_metadata.pkl')

        # log param
        mlflow.log_params(network_params)

        # log statistics
        _scaled = ''
        if network_params['connectivity_scaling']:
            _scaled = '_scaled'
        mlflow.log_artifacts(output_path)
        partner_blk1_stats = (
            task.groupby('Partner')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME]
            ].describe()
        )
        partner_blk2_stats = (
            task.groupby('Partner')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME]
            ].describe()
        )
        partner_scaled_blk1_stats = (
            task.groupby('Partner')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )
        partner_scaled_blk2_stats = (
            task.groupby('Partner')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )

        # stack all stats rowwise a single dataframe
        partner_blk1_stats.to_csv(output_path / 'partner_blk1_stats.csv')
        partner_blk2_stats.to_csv(output_path / 'partner_blk2_stats.csv')
        partner_scaled_blk1_stats.to_csv(output_path / 'partner_scaled_blk1_stats.csv')
        partner_scaled_blk2_stats.to_csv(output_path / 'partner_scaled_blk2_stats.csv')

        condition_blk1_stats = (
            task.groupby('Condition')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME]
            ].describe()
        )
        condition_blk2_stats = (
            task.groupby('Condition')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME]
            ].describe()
        )
        condition_scaled_blk1_stats = (
            task.groupby('Condition')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )
        condition_scaled_blk2_stats = (
            task.groupby('Condition')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )
        condition_blk1_stats.to_csv(output_path / 'condition_blk1_stats.csv')
        condition_blk2_stats.to_csv(output_path / 'condition_blk2_stats.csv')
        condition_scaled_blk1_stats.to_csv(output_path / 'condition_scaled_blk1_stats.csv')
        condition_scaled_blk2_stats.to_csv(output_path / 'condition_scaled_blk2_stats.csv')

        label_blk1_stats = (
            task.groupby('label')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME]
            ].describe()
        )
        label_blk2_stats = (
            task.groupby('label')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME]
            ].describe()
        )
        label_scaled_blk1_stats = (
            task.groupby('label')[
                network_params['blocks'][0] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )
        label_scaled_blk2_stats = (
            task.groupby('label')[
                network_params['blocks'][1] + '_' + network_params[CE_NAME] + _scaled
            ].describe()
        )
        label_blk1_stats.to_csv(output_path / 'label_blk1_stats.csv')
        label_blk2_stats.to_csv(output_path / 'label_blk2_stats.csv')
        label_scaled_blk1_stats.to_csv(output_path / 'label_scaled_blk1_stats.csv')
        label_scaled_blk2_stats.to_csv(output_path / 'label_scaled_blk2_stats.csv')

        mlflow.log_artifacts(output_path)
        logger.info(f'Networks run_id {active_run.info.run_id} '
                    f'exported at {str(output_path)}')


if __name__ == '__main__':
    """Launch network_generation."""
    parser = argparse.ArgumentParser()

    PATH = str(Path.cwd() / 'data')
    parser.add_argument('--connectivity_estimator',
                        type=str,
                        default='manual_threshold_num_salient_values')
    parser.add_argument('--actual_filename',
                        type=str,
                        default='connectivity/hbo_actual_task.parquet')
    parser.add_argument('--shuffled_filename',
                        type=str,
                        default='connectivity/hbo_permutation_task.parquet')
    parser.add_argument('--chromophore', type=str, default='Hbo')
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--artifact_path', type=str)

    args = parser.parse_args()

    if os.getenv('MLFLOW_TRACKING_URI', None):
        logger.info('MLFLOW_TRACKING_URI not set')

    if (args.actual_filename is None) or (args.shuffled_filename is None):
        raise Exception(
            'argument --actual_filename and --shuffled_filename should be specified'
        )

    if args.run_id and args.artifact_path:
        config = asdict(
            NetworkParams(
                chromophore=args.chromophore,
                parent_run_id=args.run_id,
                input_connectivity_estimator=args.connectivity_estimator
            )
        )

        local_path = fetch_artifacts(args.run_id, args.artifact_path)
        network_generation(args.actual_filename,
                           args.shuffled_filename,
                           config,
                           local_path=local_path)
    else:
        config = asdict(
            NetworkParams(
                chromophore=args.chromophore,
                input_connectivity_estimator=args.connectivity_estimator
            )
        )
        logger.info(NetworkParams.chromophore)
        network_generation(args.actual_filename, args.shuffled_filename, config, local_path=PATH)
