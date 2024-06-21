import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from functools import reduce

class PlasmaVolume:

    def __init__(self, data_path: str, date_column: str, experimental: str):
        self.data_path = data_path
        self.date_column = date_column
        self.__experimental = experimental
        self.update_data()


    def update_data(self):
        df = pd.read_csv(self.data_path)

        df['month'] = pd.to_datetime(df[self.date_column]).dt.to_period('M').dt.start_time.astype(str)
        df['day'] = pd.to_datetime(df[self.date_column]).dt.to_period('D').dt.start_time.astype(str)
        df['week'] = pd.to_datetime(df[self.date_column]).dt.to_period('w').dt.start_time.astype(str)

        self.total_data = df.copy()

        self.data = df[df['group'] == self.__experimental]

        self.data['persona_1'] = self.__calculate_plasma_volume(persona_1=True)
        self.data['persona_2'] = self.__calculate_plasma_volume(persona_1=False)
        self.data['overall'] = 'overall'


    def __calculate_plasma_volume(self, persona_1):

        weight = self.data['weight'].values
        height = self.data['height'].values
        hematocrit = self.data['hematocrit'].values / 100

        if persona_1:
            hyp_per_tgt = (70 / np.sqrt((weight * 0.453592 / ((height * 0.0254) ** 2)) / 22) * weight * 0.453592) * (1 - hematocrit) * 0.285
        else:
            hyp_per_tgt = (70 / np.sqrt((weight * 0.453592 / ((height * 0.0254) ** 2)) / 22) * weight * 0.453592) * (1 - hematocrit) * 0.305

        hyp_per_tgt = np.where(hyp_per_tgt > 1000, 1000, hyp_per_tgt)

        return hyp_per_tgt


    def get_plasma_collection_summary(self):

        a = self.total_data.groupby('group')['yield'].mean().reset_index().rename(columns = {'yield': 'average_yield'})
        b = self.total_data.groupby('group')['actual_vol'].sum().reset_index().rename(columns = {'actual_vol': 'total_plasma_collected'})
        c = self.total_data.groupby('group')['target_vol'].mean().reset_index().rename(columns = {'target_vol': 'average_target_colume'})
        d = self.total_data.groupby('group')['actual_vol'].mean().reset_index().rename(columns = {'actual_vol': 'average_collected_volume'})

        summary = reduce(lambda df_1, df_2: pd.merge(df_1, df_2, on = 'group'), [a, b, c, d])

        return summary


    def get_comparative_gain(self, population, stratify_by):

        data = self.data[self.data[population]]

        a = data.groupby([stratify_by])[['persona_1', 'persona_2']].apply(lambda x: [sms.DescrStatsW(x['persona_1']), sms.DescrStatsW(x['persona_2'])]).apply(lambda x: sms.CompareMeans(*x).tconfint_diff(usevar='unequal', alpha=0.05))
        b = data.groupby(['overall'])[['persona_1', 'persona_2']].apply(lambda x: [sms.DescrStatsW(x['persona_1']), sms.DescrStatsW(x['persona_2'])]).apply(lambda x: sms.CompareMeans(*x).tconfint_diff(usevar='unequal', alpha=0.05))

        stat = pd.concat([a.reset_index(), b.reset_index().rename(columns = {'overall': stratify_by})])
        stat['left'] = stat[0].apply(lambda x: x[0])
        stat['right'] = stat[0].apply(lambda x: x[1])
        stat = stat.drop([0], axis=1)

        q = data.groupby([stratify_by])[['persona_1', 'persona_2']].mean().diff(axis=1)['persona_2']
        w = data.groupby(['overall'])[['persona_1', 'persona_2']].mean().diff(axis=1)['persona_2']
        stat_mean = pd.concat([q.reset_index(), w.reset_index().rename(columns = {'overall': stratify_by})]).rename(columns = {'persona_2': 'mean'})

        stat = stat.merge(stat_mean)

        return stat


    def get_longitudinal_stat(self, population, show_by):

        data = self.data[self.data[population]]

        longitudinal_stat = data.groupby([show_by])[['persona_1', 'persona_2']].mean()
        longitudinal_stat['delta'] = longitudinal_stat.diff(axis = 1)['persona_2']
        # longitudinal_stat['delta_percentage'] = 100*longitudinal_stat['delta']/longitudinal_stat['persona_1']

        return longitudinal_stat



    def __check_data(self):

        if self.date_column not in self.data.columns:
            raise KeyError(f'Provided DATE column {self.date_column} not in the Data.')

        if self.id_column not in self.data.columns:
            raise KeyError(f'Provided ID column {self.id_column} not in the Data.')