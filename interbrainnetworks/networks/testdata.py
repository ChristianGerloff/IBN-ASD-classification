"""Creates test data"""
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from numpy.random import randint, uniform


def _create_sample(idx: int,
                   conditions: list = ['cooperation'],
                   partners: list = ['peer', 'stranger'],
                   probes: list = ['Probe_1'],
                   channels: np.ndarray = np.arange(1, 23),
                   all_idx: np.ndarray = np.arange(1, 30),
                   badchannels: bool = True,
                   seed: int = 20211001) -> list:
    """Create a sample dataframe.

    Args:
        idx (int): index of the sample
        conditions (list, optional): list of conditions.
            Defaults to ['cooperation'].
        partners (list, optional): list of partners.
            Defaults to ['peer', 'stranger'].
        probes (list, optional): list of probes. Defaults to ['Probe_1'].
        channels (np.ndarray, optional): list of channels.
            Defaults to np.arange(1, 23).
        all_idx (np.ndarray, optional): list of all indices.
            Defaults to np.arange(1, 30).
        badchannels (bool, optional): whether to exclude some bad channels.
            Defaults to True.
        seed (int, optional): seed for random number generator.
            Defaults to 20211001.

    Returns:
        list: list of dataframes
    """

    # randomly remove channels
    if badchannels > 0:
        np.random.seed(seed)
        n_badchannels = randint(1, round(len(channels) * 0.2))
        idx_badchannels = randint(0, len(channels) - 1, size=n_badchannels)
        channels = np.delete(channels, idx_badchannels)

    categories = [conditions, partners, probes, channels, channels]

    task = pd.DataFrame()
    rest = pd. DataFrame()
    task_permu = pd.DataFrame()
    rest_permu = pd. DataFrame()

    data = [(c, p, prob, ch_1, ch_2)
            for (c, p, prob, ch_1, ch_2) in product(*categories)]

    # task
    task['Block_1_salient_wco'] = uniform(0, 0.7, len(data)) * 1.01
    task['Block_2_salient_wco'] = uniform(0, 0.7, len(data)) * 1.01
    task['Condition'] = [i[0] for i in data]
    task['Partner'] = [i[1] for i in data]
    task['Probe_1'] = [i[2] for i in data]
    task['Channel_1'] = [i[3] for i in data]
    task['Channel_2'] = [i[4] for i in data]
    task['ID'] = idx
    task['Baseline'] = 'task'

    # rest
    rest['Block_0_salient_wco'] = uniform(0, 0.5, len(data)) * 1.01
    rest['Condition'] = [i[0] for i in data]
    rest['Partner'] = [i[1] for i in data]
    rest['Probe_1'] = [i[2] for i in data]
    rest['Channel_1'] = [i[3] for i in data]
    rest['Channel_2'] = [i[4] for i in data]
    rest['ID'] = idx
    rest['Baseline'] = 'baseline'

    # Simulate partner
    task.loc[task['Partner'] == partners[0], 'Block_1_salient_wco'] = (
        task.loc[task['Partner'] == partners[0], 'Block_1_salient_wco'] * 1.2
    )

    task.loc[task['Partner'] == partners[0], 'Block_2_salient_wco'] = (
        task.loc[task['Partner'] == partners[0], 'Block_2_salient_wco'] * 1.2
    )

    rest.loc[rest['Partner'] == partners[0], 'Block_0_salient_wco'] = (
        rest.loc[rest['Partner'] == partners[0], 'Block_0_salient_wco'] * 1.2
    )

    categories.append(all_idx)
    data = [(c, p, prob, ch_1, ch_2, i)
            for (c, p, prob, ch_1, ch_2, i) in product(*categories)
            if i != idx]

    # task
    task_permu['Block_1_salient_wco'] = uniform(0, 0.7, len(data))
    task_permu['Block_2_salient_wco'] = uniform(0, 0.6, len(data))
    task_permu['Condition'] = [i[0] for i in data]
    task_permu['Partner'] = [i[1] for i in data]
    task_permu['Probe_1'] = [i[2] for i in data]
    task_permu['Channel_1'] = [i[3] for i in data]
    task_permu['Channel_2'] = [i[4] for i in data]
    task_permu['ID_1'] = idx
    task_permu['ID_2'] = [i[5] for i in data]
    task_permu['Baseline'] = 'task'

    # rest
    rest_permu['Block_0_salient_wco'] = uniform(0, 0.5, len(data))
    rest_permu['Condition'] = [i[0] for i in data]
    rest_permu['Partner'] = [i[1] for i in data]
    rest_permu['Probe_1'] = [i[2] for i in data]
    rest_permu['Channel_1'] = [i[3] for i in data]
    rest_permu['Channel_2'] = [i[4] for i in data]
    rest_permu['ID_1'] = idx
    rest_permu['ID_2'] = [i[5] for i in data]
    rest_permu['Baseline'] = 'baseline'

    return task, rest, task_permu, rest_permu


def create_testdata(n_samples: int, path: Path = None):
    """create a dataset of n_samples and
        save dataframes as parquet files.

    Args:
        n_samples (int): number of samples
        path (Path): path to save the dataframes.
            Defaults to None.

    Returns:
        : filenames of tasks, rests, task_permutations, rest_permutations
    """

    idx = np.arange(1, n_samples)
    sets = [_create_sample(i, all_idx=idx) for i in idx]
    task = pd.concat([s[0] for s in sets])
    rest = pd.concat([s[1] for s in sets])
    task_permu = pd.concat([s[2] for s in sets])
    rest_permu = pd.concat([s[3] for s in sets])

    task_filename = path / 'task.parquet'
    rest_filename = path / 'rest.parquet'
    task_permu_filename = path / 'task_permutation.parquet'
    rest_permu_filename = path / 'rest_permutation.parquet'

    filenames = (
        task_filename, rest_filename, task_permu_filename, rest_permu_filename
    )

    path.mkdir(parents=True, exist_ok=True)
    task.to_parquet(task_filename)
    rest.to_parquet(rest_filename)
    task_permu.to_parquet(task_permu_filename)
    rest_permu.to_parquet(rest_permu_filename)

    return filenames
