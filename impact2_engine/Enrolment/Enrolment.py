import pandas as pd
import pyreadr
import datetime
from dateutil.relativedelta import relativedelta, MO
import numpy as np
import json
from functools import reduce
from impact2_engine import MissingDataException
from prophet import Prophet


class Enrolment:

    __POSSIBLE_METRICS = {'min', 'max', 'mean', 'median', 'p_25', 'p_75'}

    def __init__(self, data_path: str, date_column: str, id_column: str):
        self.data_path = data_path
        self.date_column = date_column
        self.id_column = id_column
        self.read_data()
        self.__check_data()
        self.preprocess_data()


    def read_data(self):
        df = pyreadr.read_r(self.data_path)[None]

        # TODO: there should be no NAs in the data.
        # What to do if there are NAs
        df = df.dropna()

        if df.isna().sum().sum() != 0:
            raise MissingDataException('Data contains NAs.')

        self.data = df


    def preprocess_data(self):
        self.data['week'] = self.data[self.date_column].apply(lambda x: x + relativedelta(weekday=MO(-1)))


    def calculate_working_data(self, columns_to_use: list, donor_level: bool) -> pd.DataFrame:

        if donor_level:
            to_work = self.data.groupby(columns_to_use + ['week'] )[[self.id_column]].nunique().reset_index()
        else:
            to_work = self.data.groupby(columns_to_use + ['week'] )[[self.id_column]].count().reset_index()

        to_work_wide = pd.pivot(to_work, columns = columns_to_use, index=['week'])

        return to_work_wide


    def get_enrolment_rate_metrics(self, site: str, strata: list, donor_level: bool, list_of_metrics: list) -> pd.DataFrame:

        columns_to_use = [site] + strata

        df = self.calculate_working_data(columns_to_use, donor_level)

        metrics = self.__calculate_metrics(df, list_of_metrics)

        donation_p = self.data.groupby(columns_to_use)[self.id_column].count()/ self.data.groupby([site])[self.id_column].count()
        donation_p = donation_p.reset_index().rename({self.id_column: 'donation_percentage'}, axis=1)
        metrics.append(donation_p)
        donation_c = self.data.groupby(columns_to_use)[self.id_column].count().reset_index().rename({self.id_column: 'donation_count'}, axis=1)
        metrics.append(donation_c)

        donor_p = self.data.groupby(columns_to_use)[self.id_column].nunique()/ self.data.groupby([site])[self.id_column].nunique()
        donor_p = donor_p.reset_index().rename({self.id_column: 'donor_percentage'}, axis=1)
        metrics.append(donor_p)
        donor_c = self.data.groupby(columns_to_use)[self.id_column].nunique().reset_index().rename({self.id_column: 'donor_count'}, axis=1)
        metrics.append(donor_c)

        if len(metrics) == 1:
            return metrics[0]
        else:
            return reduce(lambda df_1, df_2: pd.merge(df_1, df_2, on = columns_to_use), metrics)


    def proj(self, df: list, increment: float):

        return df.fillna(0).values[-1] + increment


    def get_projections(self, max_weeks: int, target_sample: int, columns_to_use: str, donor_level: bool, projection_steps: dict):

        projection_steps = pd.DataFrame(projection_steps).groupby(columns_to_use).sum()['proj_value'].to_dict()
        to_work = self.calculate_working_data(columns_to_use, donor_level)
        to_work = to_work.cumsum()
        to_work = to_work.droplevel(0, axis=1)
        to_work = to_work.melt(ignore_index=False).reset_index()

        to_work_dict = {}

        projected_sample = 0
        new_week_count = 0

        for name, group in to_work.groupby(columns_to_use):
            to_work_dict[name] = group
            projected_sample += np.sum(group['value'].fillna(0).values[-1])

        while (new_week_count < max_weeks) & (projected_sample < target_sample):
            for name in to_work_dict.keys():
                temp_projection = self.proj(to_work_dict[name]['value'], projection_steps[name])
                projected_sample += temp_projection

                next_week = to_work_dict[name]['week'].max() + relativedelta(weekday=MO(2))

                temp_projection_df = pd.DataFrame.from_records([{'week': next_week, 'value': temp_projection}])
                to_work_dict[name] = pd.concat([to_work_dict[name], temp_projection_df])

            new_week_count += 1

        temp_df_list = []

        for key in to_work_dict.keys():
            temp = to_work_dict[key]

            temp[columns_to_use] = temp[columns_to_use].ffill()
            temp = temp.fillna(0)

            temp_df_list.append(temp)

        temp_df = pd.concat(temp_df_list).reset_index(drop=True)

        return temp_df



    def get_bar_chart(self, show_by: str, by_donor: bool, stratify_by: list, only_new_donors: bool=False):

        if only_new_donors:
            temp_df = self.data.sort_values(by=self.date_column, ascending=True)
            temp_df = temp_df.drop_duplicates(keep='first')
        else:
            temp_df = self.data


        if show_by == 'week':
            sort_by = 'week'
            temp_df[sort_by] = temp_df[sort_by].apply(lambda x: x.normalize())
        else:
            sort_by = 'day'
            temp_df['day'] = temp_df[self.date_column].apply(lambda x: x.normalize())


        if not by_donor:
            result = temp_df.groupby([sort_by] + stratify_by)[self.id_column].count().reset_index()
            result['cumsum'] = result.reset_index().groupby(stratify_by)[self.id_column].cumsum()
        else:
            result = temp_df.groupby([sort_by] + stratify_by)[self.id_column].nunique().reset_index()
            result['cumsum'] = result.reset_index().groupby(stratify_by)[self.id_column].cumsum()


        percentage = result.groupby(sort_by)[self.id_column].sum().reset_index()
        percentage.columns = [sort_by, 'total']

        result = pd.merge(result, percentage, on=sort_by)

        result['percentage'] = 100*result[self.id_column]/result['total']

        result[sort_by] = result[sort_by].astype(str)

        result = result.rename({self.id_column : 'absolute'}, axis = 1)
        result = result.drop(['total'], axis = 1)
        
        return result


    def get_model_projections(self, max_weeks: int, target_sample: int, columns_to_use: list, donor_level: bool):

        df = self.data.copy()

        df['day'] = df[self.date_column].apply(lambda x: x.normalize())

        df_preds = []

        for group,j in df.groupby(columns_to_use):

            if donor_level:
                df_train = j.groupby('day')[self.id_column].nunique().reset_index()
            else:
                df_train = j.groupby('day')[self.id_column].count().reset_index()

            orig_columns = df_train.columns
            df_train.columns = ['ds', 'y']
            df_train['y'] = df_train['y'].cumsum()
            m = Prophet(changepoint_prior_scale=0.5)
            m.fit(df_train)

            preds = m.make_future_dataframe(max_weeks)
            preds = m.predict(preds)

            for col, value in zip(columns_to_use, group):
                preds[col] = value

            preds = preds[['ds', 'yhat_lower', 'yhat_upper', 'yhat'] + columns_to_use].reset_index(drop=True)

            df_preds.append(preds)

        df_preds = pd.concat(df_preds)
        dtime = df_preds.groupby('ds')['yhat'].sum()[df_preds.groupby('ds')['yhat'].sum() >= target_sample]
        if dtime.shape[0] > 0:
            dtime = dtime.index[0]
            df_preds = df_preds[df_preds['ds'] <= dtime]
        df_preds = df_preds.rename(columns={m: n for m, n in zip(['ds', 'y'], orig_columns)})

        return df_preds


    def get_activation_date(self, site: str):
        
        activation_date = self.data.groupby(site)[self.date_column].min().reset_index().rename(columns = {self.date_column: 'activation_date'})
        activation_date['activation_date'] = activation_date['activation_date'].apply(lambda x: x.normalize()).astype(str)

        return activation_date


    def __calculate_metrics(self, df: pd.DataFrame, list_of_metrics: list) -> pd.DataFrame:

        if not list_of_metrics:
            raise KeyError(f'Variable list_of_metrics cannot be empty. Valid METRICs are {Enrolment.__POSSIBLE_METRICS}.')

        if set(list_of_metrics).difference(Enrolment.__POSSIBLE_METRICS):
            raise KeyError(f'Provided METRIC method {set(list_of_metrics).difference(Enrolment.__POSSIBLE_METRICS)} is not valid. Valid METRICs are {Enrolment.__POSSIBLE_METRICS}.')

        temp_df = []

        if 'min' in list_of_metrics:
            min_ = df.apply(np.min).reset_index().drop(['level_0'], axis=1).rename({0: 'min'}, axis=1)
            temp_df.append(min_)
        if 'max' in list_of_metrics:
            max_ = df.apply(np.max).reset_index().drop(['level_0'], axis=1).rename({0: 'max'}, axis=1)
            temp_df.append(max_)

        if 'mean' in list_of_metrics:
            mean_ = df.apply(np.mean).reset_index().drop(['level_0'], axis=1).rename({0: 'mean'}, axis=1)
            temp_df.append(mean_)
        if 'median' in list_of_metrics:
            median_ = df.apply(np.nanmedian).reset_index().drop(['level_0'], axis=1).rename({0: 'median'}, axis=1)
            temp_df.append(median_)

        if 'p_25' in  list_of_metrics:
            p_25 = df.apply(lambda x: np.nanpercentile(x, 25)).reset_index().drop(['level_0'], axis=1).rename({0: 'p_25'}, axis=1)
            temp_df.append(p_25)
        if 'p_75' in list_of_metrics:
            p_75 = df.apply(lambda x: np.nanpercentile(x, 75)).reset_index().drop(['level_0'], axis=1).rename({0: 'p_75'}, axis=1)
            temp_df.append(p_75)

        return temp_df


    def __check_data(self):

        if self.date_column not in self.data.columns:
            raise KeyError(f'Provided DATE column {self.date_column} not in the Data.')

        if self.id_column not in self.data.columns:
            raise KeyError(f'Provided ID column {self.id_column} not in the Data.')