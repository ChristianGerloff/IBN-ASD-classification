"""
Generates object for inter-brain network related data
contains specific data wrangling.
"""
import pandas as pd
import atlasreader.atlasreader as ar

from logging import info, warning

SEPERATOR = '_'
OFFSET_X = 200  # positional offset to avoid overlapping node sets


class Networks(object):

    def __init__(self,
                 dyad_type: str,
                 chromophore: str = None,
                 atlas: str = 'default',
                 channel_set: int = 22,
                 input_connectivity_estimator: str = None,
                 **kwargs):
        """Initializes inter-brain networks object.

        Args:
            dyad_type (str): specifies the type of dyad to be used.
            chromophore (str, optional): specifies the chromophore of
                the dataset. Defaults to None.
            atlas (str, optional): specifies the atlas to be used.
                Defaults to 'default'.
            channel_set (int, optional): specifies the channel set to be used.
                Defaults to 22.
            input_connectivity_estimator (str, optional):
                specifies the input connectivity estimator to be used.
                Defaults to None.
        """

        self.dyad_type = dyad_type
        self.chromophore = chromophore
        self.atlas = atlas
        self.channel_set = channel_set
        self.input_connectivity_estimator = input_connectivity_estimator
        self._set_obj_params(**kwargs)

    def _set_obj_params(self, **parameters):
        """Sets object parameters. """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    @property
    def block_estimators(self):
        """Returns list of block estimators. """

        # specify estimators depending on block design
        if (hasattr(self, 'blocks') and
           self.blocks is not None):
            self._estimators = []
            for i in self.blocks:
                self._estimators = (
                   self._estimators +
                   [i + SEPERATOR + self.input_connectivity_estimator]
                )
        else:
            self._estimators = [
                'Block_0' + SEPERATOR + self.input_connectivity_estimator,
                self.input_connectivity_estimator
            ]
        return self._estimators

    @property
    def block_estimators_scaled(self):
        """Returns list of scaled block estimators. """
        if (any(self.factors_scaling) and
           hasattr(self, 'connectivity_scaling') and
           self.connectivity_scaling):
            self._estimators_scaled = [
                s + '_scaled' for s in self.block_estimators
            ]
        else:
            self._estimators_scaled = self.block_estimators
            warning('Mean scaling was deactivated. Values will be unscaled')
        return self._estimators_scaled

    @property
    def mni(self):
        """Returns dictionary of MNI coordinates. """
        return self._mni

    @mni.setter
    def mni(self, coordinates: pd.DataFrame) -> dict:
        """Sets MNI coordinates and assigns areas to channel locations.

        Args:
            coordinates (pd.DataFrame): coordinates of channel locations.

        Returns:
            dict: dictionary of MNI coordinates.
        """
        if (self.atlas is not None and
           self.atlas != 'default'):
            coordinates['Area'] = (
                coordinates.apply(
                    lambda row: ar.read_atlas_peak(
                        ar.get_atlas(self.atlas),
                        [row['X'], row['Y'], row['Z']]),
                    axis=1)
            )

        coordinates.drop(columns=['Area'])
        self._mni = coordinates.to_dict()

    @property
    def excluded_channels(self) -> int:
        """Returns percentage of excluded channels."""
        groups_all_edges = ['ID', 'Baseline',
                            'Condition', 'Partner', 'Probe_1']
        excluded_combinations = (
            self.data.groupby(groups_all_edges)['channel_pair'].count()
        )

        num_excluded = sum((self.channel_set**2) - excluded_combinations)
        todal_pos = len(excluded_combinations) * (self.channel_set**2)
        self._perc = 100 * num_excluded / todal_pos
        info(f'Excluded channel combinations {num_excluded} '
             f'out of {todal_pos}: {self._perc}%')
        return self._perc

    @property
    def scaling_factor(self) -> pd.DataFrame:
        """Calculates mean acorss all block estimators."""

        if hasattr(self, 'factors_scaling'):
            if self.separated_estimators:
                self._mean_estimators = (
                    self.data.groupby(self.factors_scaling)[
                        self.block_estimators].apply(lambda r: r.mean(axis=0))
                )
                self._mean_estimators.reset_index(inplace=True)
            else:
                self._mean_estimators = (
                    self.data.groupby(self.factors_scaling)[
                        self.block_estimators].apply(
                            lambda r: r.mean(axis=0)).mean(axis=1)
                )
                self._mean_estimators = (
                    self._mean_estimators.to_frame('scaling_factor')
                )
                self._mean_estimators.reset_index(inplace=True)
        else:
            self._mean_estimators = 0
            warning('No scaling factors specified.')
        return self._mean_estimators

    def scale(self, scale_mean: pd.DataFrame = None):
        """Calculates mean acorss all scaled block estimators

        Args:
            scale_mean (pd.DataFrame, optional): mean scaling factors.
                Defaults to None.
        """
        # test if data where set
        if hasattr(self, 'data') is False:
            raise ValueError('Data were not specified.')

        if scale_mean is None:
            scale_mean = self.scaling_factor

        data = self.data

        # set mean for blocks seperately
        if self.connectivity_scaling and self.separated_estimators:

            # ensure that mean was not set before
            if (hasattr(self, 'connectivity_scaling') and
               set(self.block_estimators_scaled).issubset(data.columns)):
                data = data.drop(columns=self.block_estimators_scaled)

            # add mean and calculate scaled value of estimators
            data = data.merge(scale_mean,
                              how='inner',
                              on=self.factors_scaling,
                              sort=False,
                              suffixes=('', '_scaled'),
                              copy=False)

            query = zip(
                *[data[e] - data[e_s] for e, e_s in zip(
                    self.block_estimators, self.block_estimators_scaled)]
            )

            data[self.block_estimators_scaled] = pd.DataFrame(query)

        elif self.connectivity_scaling and self.separated_estimators is False:
            # ensure that mean was not set before
            if any('scaling_factor' in c for c in data.columns):
                data = data.drop(columns='scaling_factor')

            # add mean and calculate scaled value of estimators
            data = data.merge(scale_mean,
                              how='inner',
                              on=self.factors_scaling,
                              sort=False,
                              copy=False)

            query = zip(*[data[e] - data['scaling_factor']
                        for e in self.block_estimators])

            data[self.block_estimators_scaled] = pd.DataFrame(query)

        else:
            info('Mean scaling was deactivated. Values will be unscaled')

            # ensure that mean was not set before
            data['scaling_factor'] = 0

        self.data = data

    def transform(self,
                  permu_set: pd.DataFrame,
                  scaling_factors: pd.DataFrame = None,
                  null_dist: pd.DataFrame = None) -> pd.DataFrame:
        """Calculates null distribution. Here in a group wise manner.

        Args:
            permu_set (pd.DataFrame): permutation set
            scaling_factors (pd.DataFrame, optional): scales data.
                Defaults to None.
            null_dist (pd.DataFrame, optional): provided null distribution.
                Defaults to None.

        Returns:
            pd.DataFrame: null distribution.
        """

        # test if data where set
        if hasattr(self, 'data') is False:
            raise ValueError('Data were not specified.')

        # ensure that exchangeables are specified
        if hasattr(self, 'exchangeables') is False:
            raise ValueError('No exchangeables specified.')

        # ensure that alpha was sepcified
        if hasattr(self, 'alpha') is False:
            raise ValueError('No exchangeables specified.')

        # performs scaling if specified
        if scaling_factors is not None:
            self.scale(scaling_factors)

        data = self.data
        estimator_list_null_dist = [
            s + '_null_dist' for s in self.block_estimators_scaled]

        if (null_dist is None and self.separated_estimators):
            null_dist = (
                permu_set.groupby(
                    self.exchangeables)[self.block_estimators_scaled].apply(
                        lambda c: c.quantile(1 - self.alpha))
            )
            null_dist = null_dist.reset_index(self.exchangeables)

            data = data.merge(null_dist,
                              on=self.exchangeables,
                              sort=False,
                              suffixes=('', '_null_dist'),
                              copy=False)

            query = zip(*[data[e].fillna(0) >= data[e_s].fillna(0)
                          for e, e_s in zip(
                           self.block_estimators_scaled,
                           estimator_list_null_dist)])
            data[estimator_list_null_dist] = pd.DataFrame(query)

        # calculate quantiles across block estimators
        elif (null_dist is None and self.separated_estimators is False):
            null_dist = pd.wide_to_long(
                permu_set[
                    ['ID', 'Baseline',
                     'Condition', 'Partner', 'Probe_1',
                     'channel_pair'] +
                    self.block_estimators_scaled
                ],
                stubnames='Block',
                i=['ID', 'Baseline',
                   'Condition', 'Partner', 'Probe_1',
                   'channel_pair'],
                j='name',
                sep='_',
                suffix='(\d+|\w+)'
            )
            null_dist.reset_index(self.exchangeables, inplace=True)
            null_dist = null_dist.reset_index('name', drop=True).groupby(
                self.exchangeables).quantile(1 - self.alpha)
            null_dist = null_dist.reset_index(self.exchangeables)
            null_dist = null_dist.rename(
                columns={'Block': 'quantile_null_dist'})

            # ensure that column was not set before
            if any('quantile_null_dist' in c for c in data.columns):
                data = data.drop(columns='quantile_null_dist')
            data = data.merge(
                null_dist, on=self.exchangeables, sort=False, copy=False
            )

            query = zip(*[
                data[e].fillna(0) >= data['quantile_null_dist'].fillna(0)
                for e in self.block_estimators_scaled
            ])
            data[estimator_list_null_dist] = pd.DataFrame(query)

        elif null_dist is not None:
            data = data.merge(null_dist,
                              on=self.exchangeables,
                              sort=False,
                              suffixes=('', '_null_dist'),
                              copy=False)

            query = zip(*[data[e].fillna(0) >= data[e_s].fillna(0)
                          for e, e_s in zip(
                           self.block_estimators_scaled,
                           estimator_list_null_dist)])
            data[estimator_list_null_dist] = pd.DataFrame(query)
        else:
            exit
        self._null_dist_threshold = null_dist
        self.data = data
        return

    def set_data(self,
                 task_estimators: pd.DataFrame,
                 rest_estimators: pd.DataFrame,
                 mni: pd.DataFrame):
        """Sets data and performs preprocessing.

        Args:
            task_estimators (pd.DataFrame): task related data
            rest_estimators (pd.DataFrame): rest related data
            mni (pd.DataFrame): mni coordinates
        """
        # set mni
        self.mni = mni

        # ensure conditions are set correctly
        task_estimators['Baseline'] = 'task'
        if rest_estimators is not None:
            rest_estimators['Baseline'] = 'baseline'
            rest_estimators['Condition'] = 'rest'
            estimators = pd.concat([task_estimators, rest_estimators])
            estimators = estimators.reset_index(drop=True)
        else:
            estimators = task_estimators

        # add chromophore
        if 'chromophore' not in estimators.columns:
            estimators['chromophore'] = self.chromophore
        elif (hasattr(self, 'chromophore') and self.chromophore is not None):
            n_unfiltered = estimators.shape[0]
            estimators = (
                estimators[estimators['chromophore'] == self.chromophore]
            )
            info(f'Chromophore filtered estimators: {estimators.shape[0]}'
                 f'({n_unfiltered})')

        # filter condition
        if (hasattr(self, 'condition_filter') and
           self.condition_filter is not None and
           any(self.condition_filter)):
            n_unfiltered = estimators.shape[0]
            rel_condition = estimators.Condition.isin(self.condition_filter)
            estimators = estimators.loc[rel_condition, :]
            info(f'Condition filtered estimators: {estimators.shape[0]}'
                 f'({n_unfiltered})')

        # filter partner
        if (hasattr(self, 'partner_filter') and
           self.partner_filter is not None and
           any(self.partner_filter)):
            n_unfiltered = estimators.shape[0]
            estimators = (
                estimators.loc[
                    estimators.Partner.isin(self.partner_filter), :]
                )
            info(f'Partner filtered estimators: {estimators.shape[0]}'
                 f'({n_unfiltered})')

        # filter ID
        if (hasattr(self, 'id_filter') and
           self.id_filter is not None and
           any(self.id_filter)):
            n_unfiltered = estimators.shape[0]
            if self.dyad_type == 'permu':
                estimators = estimators.loc[
                    ~estimators.ID_1.isin(self.id_filter), :]
                estimators = estimators.loc[
                    ~estimators.ID_2.isin(self.id_filter), :]
            else:
                estimators = estimators.loc[
                    ~estimators.ID.isin(self.id_filter), :]
            info(f'Id filtered estimators: {estimators.shape[0]}'
                 f'({n_unfiltered})')

        self.data = estimators
        self.__preprocessing()

    def __preprocessing(self):
        """Prepare data network generation."""
        estimators = self.data.copy()

        # invert and scale shannon entroy
        for i in self.block_estimators:
            if i.__contains__('shannon'):
                estimators[i] = 1 / estimators[i].abs()
                info(f'{i} was inverted')

        # invert shannon entroy
        for i in self.block_estimators:
            if i.__contains__('plv'):
                estimators[i] = 1 - estimators[i].abs()
                info(f'{i} was inverted')

        # consider different structure of actual and shuffled pairs
        if self.dyad_type == 'actual':
            estimators.loc[:, 'signal_pair'] = (
                estimators['ID'].astype(str) + SEPERATOR +
                estimators['Probe_1'].astype(str) + SEPERATOR +
                estimators['Baseline'].astype(str) + SEPERATOR +
                estimators['Condition'].astype(str) + SEPERATOR +
                estimators['Partner'].astype(str)
            )

            # rename baseline to rest
            test_baseline = len(
                estimators.loc[estimators['Baseline'] == 'baseline',
                               'Condition'].unique()
            )
            if test_baseline == 1:
                estimators.loc[estimators['Baseline'] == 'baseline',
                               'Baseline'] = 'rest'

        else:

            estimators['ID'] = (
                estimators['ID_1'].astype(str) + SEPERATOR +
                estimators['ID_2'].astype(str)
            )

            estimators['signal_pair'] = (
                estimators['ID_1'].astype(str) + SEPERATOR +
                estimators['ID_2'].astype(str) + SEPERATOR +
                estimators['Probe_1'].astype(str) + SEPERATOR +
                estimators['Baseline'].astype(str) + SEPERATOR +
                estimators['Condition'].astype(str) + SEPERATOR +
                estimators['Partner'].astype(str)
            )

            # factorize
            (estimators['ID_1'],
             self.id_1_labels) = pd.factorize(estimators['ID_1'])
            (estimators['ID_2'],
             self.id_2_labels) = pd.factorize(estimators['ID_2'])

        # factorize
        estimators['ID'], self.id_labels = pd.factorize(estimators['ID'])
        estimators['signal_pair'], self.signal_pair_labels = (
            pd.factorize(estimators['signal_pair'])
        )
        estimators['conditions'] = (
            estimators['Condition'].astype(str) + SEPERATOR +
            estimators['Partner'].astype(str)
        )

        # set mni coordinates
        estimators['Channel_1_X'] = estimators['Channel_1'].map(self.mni['X'])
        estimators['Channel_1_Y'] = estimators['Channel_1'].map(self.mni['Y'])
        estimators['Channel_1_Z'] = estimators['Channel_1'].map(self.mni['Z'])
        estimators['Area_1'] = estimators['Channel_1'].map(self.mni['Area'])
        estimators['Channel_2_X'] = estimators['Channel_2'].map(self.mni['X'])
        estimators['Channel_2_Y'] = estimators['Channel_2'].map(self.mni['Y'])
        estimators['Channel_2_Z'] = estimators['Channel_2'].map(self.mni['Z'])
        estimators['Area_2'] = estimators['Channel_2'].map(self.mni['Area'])

        # Channel prefix to avoid node duplicates
        estimators['Channel_1'] = estimators['Channel_1'].astype(int)
        estimators['Channel_1'] = (
            'C' + SEPERATOR + estimators['Channel_1'].astype(str)
        )

        estimators['Channel_2'] = estimators['Channel_2'].astype(int)
        estimators['Channel_2'] = (
            'P' + SEPERATOR + estimators['Channel_2'].astype(str)
        )

        # coordinate dictionary
        coordinate_cols = ['Channel_1_X', 'Channel_1_Y', 'Channel_1_Z']
        coordinates_1 = (
            estimators.groupby('Channel_1')[coordinate_cols].agg(['unique'])
        )
        coordinates_1['set'] = 1
        coordinates_1.rename(columns={'Channel_1_X': 'X',
                                      'Channel_1_Y': 'Y',
                                      'Channel_1_Z': 'Z'},
                             inplace=True)

        coordinate_cols = ['Channel_2_X', 'Channel_2_Y', 'Channel_2_Z']
        coordinates_2 = (
            estimators.groupby('Channel_2')[coordinate_cols].agg(['unique'])
        )
        coordinates_2['set'] = 2
        coordinates_2.rename(columns={'Channel_2_X': 'X',
                                      'Channel_2_Y': 'Y',
                                      'Channel_2_Z': 'Z'},
                             inplace=True)
        coordinates = pd.concat([coordinates_1, coordinates_2])
        self.coordinates = coordinates

        # add offset
        estimators['Channel_1_X'] = estimators['Channel_1_X'] + OFFSET_X

        estimators['channel_pair'] = (
            estimators['Channel_1'].astype(str) + SEPERATOR +
            estimators['Channel_2'].astype(str)
        )

        estimators['Area'] = (
            estimators['Area_1'].astype(str) + SEPERATOR +
            estimators['Area_2'].astype(str)
        )

        # Insert Count of possible edges
        groups_all_edges = ['ID', 'Baseline',
                            'Condition', 'Partner', 'Probe_1']
        groups_area_edges = ['ID', 'Baseline',
                             'Condition', 'Partner', 'Probe_1', 'Area']
        estimators['n_possible_edges'] = (
            estimators.groupby(
                groups_all_edges)['channel_pair'].transform('count')
        )
        estimators['n_area_edges'] = (
            estimators.groupby(
                groups_area_edges)['channel_pair'].transform('count')
        )

        self.data = estimators
