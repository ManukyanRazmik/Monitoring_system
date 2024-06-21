"""Plasma Collection monitoring module for IMPACT study"""

# %% Import libraries

from typing import Optional, Union, Callable, Any
from warnings import warn
import datetime as dt
import numpy as np
import pandas as pd
from impact2_engine.utils.utils import quantile_025, quantile_975, raw_data


# %% Plasma Collection class and methods definition


class PlasmaCollection:
    """Plasma Collection monitoring module
    """

    __metrics: dict[str, Callable[..., Any]] = {
        'mean': np.mean,
        'sd': np.std,
        'median': np.median,
        'quantile_025': quantile_025,
        'quantile_975': quantile_975
    }

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.missing: Optional[pd.DataFrame] = None
        self.data = pd.DataFrame()
        self.update_data()


    def update_data(self) -> None:
        """Check, load and preprocess data at request"""

        self.__check_contents()
        self.__load_data(na_filter = self.na_filter)

        if self.na_filter:
            self.__handle_na()

        # self.__discretize_demo()
        # self.__preprocess_data()
        self.__arrange()
        self.__enrich_levels()


    def __check_contents(self) -> None:

        try:
            with open(self.data_path, 'r', encoding = 'utf-8') as csv:
                header = csv.readline().rstrip().split(',')
        except ValueError:
            print("""Ensure path is correct and contains 'utf-8' csv-file,
                  with ',' delimeter and header on the 1st line.""")

        names = [col['var'] for cols in self.contents.values()
                 for col in cols]
        absent = [name for name in names if name not in header]
        duplicate = [col for col in header if header.count(col) > 1]

        if len(absent) > 0:
            raise KeyError(
                f"Columns {', '.join(absent)} are required but absent."
            )

        if len(duplicate) > 0:
            raise KeyError(
                f"Columns {', '.join(duplicate)} are not unique."
            )

        for col in self.contents['SEV']:
            col['aes'] = [event for event in col['aes'] if event in header]


    def __load_data(self, na_filter: bool = True) -> None:

        key_names = {key: [col['var'] for col in cols]
                     for key, cols in self.contents.items()}
        col_names = [name for names in key_names.values() for name in names]
        bool_flags = key_names['POP'] + key_names['SEV']
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            dict.fromkeys(key_names['VOL'], 'float') |
            dict.fromkeys(key_names['IDS'] + key_names['DTS'], 'object') |
            dict.fromkeys(bool_flags, 'bool')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names,
            dtype = col_types,
            na_values = {col['var']: col['na']
                         for col in self.contents['VOL']
                         if 'na' in col},
            na_filter = na_filter  # detect missing value markers if True
        )

        # self.data.replace(  # convert to bool
        #     to_replace = dict.fromkeys(
        #         bool_flags,
        #         {'Yes': True, 'No': False, np.nan: False}
        #     ),
        #     inplace = True
        # )

        for col in self.contents['DTS']:  # faster than parse_dates
            self.data[col['var']] = pd.to_datetime(
                self.data[col['var']], format = col['format']
            )

        # name_to_var = {col['name']: col['var']
        #                for cols in self.contents.values()
        #                for col in cols}
        # self.data.columns = [name_to_var.get(item, item)
        #                      for item in self.data.columns]


    def __handle_na(self) -> None:

        if self.data.isnull().values.any():
            warn(
                """Found missing values. Keep only compelete cases.
                Inspect 'missing' field to find the source of NA.
                Disable na_filter to read the original data."""
            )
            self.missing = self.data[self.data.isna().any(axis = 1)]
            self.data.dropna(axis = 0, how = 'any', inplace = True)


    def __arrange(self,
                  var: Optional[str] = None,
                  reset: bool = False) -> None:

        if var is None:
            var = self.contents['DTS'][0]['var']  # provided no NA

        self.data.set_index(var, inplace = True)
        self.data.sort_index(inplace = True)

        if reset:
            self.data.reset_index(inplace = True)


    def __surrogate(self, qry: dict[str, str]) -> dict[str, Any]:

        surr_qry: dict[str, Any] = qry.copy()

        for var, lvl in qry.items():
            if (var in self.surr_lvls.keys() and
                    lvl in self.surr_lvls[var]):
                surr_qry[lvl] = True
                del surr_qry[var]

        return surr_qry

    def __enrich_levels(self) -> None:

        for col in self.contents['CAT']:
            col['lvl'] = list(self.data[col['var']].dtype.categories)

        self.strat_lvls = {col['var']: col['lvl']
                           for key, cols in self.contents.items()
                           for col in cols
                           if key in ['CAT', 'DEM']}

        self.surr_lvls = {key: [col['var'] for col in cols]
                          for key, cols in self.contents.items()
                          if key in ['POP', 'SEV']}

        self.comb_lvls = self.strat_lvls | self.surr_lvls


    def filter(self,
               start: Optional[Union[dt.date, str]] = None,
               end: Optional[Union[dt.date, str]] = None,
               query: Optional[dict[str, str]] = None) -> pd.DataFrame:
        """Slice rows based on logical query and dates range.

        Args:
            start (Optional[Union[dt.date, str]], optional): Lower bound. Defaults to None.
            end (Optional[Union[dt.date, str]], optional): Upper bound. Defaults to None.
            query (Optional[dict[str, str]], optional): Factor-level key-pair. Defaults to None.

        Returns:
            pd.DataFrame: Filtered dataframe.
        """

        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        if (start is not None and start < self.data.index[0] or
            end is not None and end > self.data.index[-1]):
            raise ValueError(
                f"""'start' and/or 'end' should be within range
                {list(self.data.index[[0, -1]].strftime('%Y-%m-%d'))}
                """
            )

        range_data = self.data.loc[slice(start, end)]

        if query is None:
            return range_data

        wrong_qry = {var: lvl for var, lvl in query.items()
                        if (var not in self.comb_lvls.keys() or
                            lvl not in self.comb_lvls[var])}

        if len(wrong_qry) > 0:
            raise ValueError(
                f"""The following pairs do not exist {str(wrong_qry)}.
                Inspect 'contents' for available levels of 'CAT' and 'DEM'.
                'POP' and 'SEV' should select the values of 'var' field."""
            )

        surr_qry = self.__surrogate(query)

        select_surr: str = ' & '.join(
            [f"({var} == '{lvl}')" if var in self.strat_lvls.keys()
                else f"({var} == {lvl})" for var, lvl in surr_qry.items()]
        )
        query_data = range_data.query(select_surr)

        return query_data



    def count_summary(self,
                      strata: Optional[list[str]] = None,
                      pop: str = 'itt',
                      raw: bool = False) -> pd.DataFrame:
        """Calculate summary for count-type metrics, plus numeric derived quantity.

        Args:
            start (Optional[Union[dt.date, str]], optional): Lower bound. Defaults to None.
            end (Optional[Union[dt.date, str]], optional): Upper bound. Defaults to None.
            strata (Optional[list[str]], optional): Variables to split by. Defaults to None.
            pop (str, optional): Population. Defaults to 'itt'.
            raw (bool, optional): Include raw data. Defaults to False.

        Returns:
            pd.DataFrame: Summary data in the longer format.
        """

        count_smry = dict.fromkeys(['col_id', 'dev_id'], [pd.Series.nunique])
        num_smry = PlasmaCollection.__metrics.copy()

        if raw:
            num_smry['raw_data'] = raw_data

        data: pd.DataFrame = self.filter(query = {'POP': pop})

        if strata is not None:
            data = data.groupby(strata)

        smry_df: pd.DataFrame = data.agg(count_smry)

        if strata is not None:
            for var in count_smry:
                smry_df[(var, 'pct_total')] = smry_df[
                    (var, 'nunique')
                ].transform(
                    lambda x: x / x.sum() * 100
                ).replace(
                    {np.nan: None}
                )

        else:
            strata = []
            smry_vec: pd.Series = smry_df.unstack().dropna()

        wkly_per_dev = self.filter(query = {'POP': pop}).groupby(
            strata + ['dev_id', 'week']
        )
        wkly_per_dev = wkly_per_dev['col_id'].nunique()
        wkly_per_dev = wkly_per_dev.groupby(strata + ['dev_id']).mean()

        if len(strata) > 0:
            wkly_per_dev = wkly_per_dev.to_frame('col_weekly_per_device')
            wkly_per_dev = wkly_per_dev.groupby(strata)
            wkly_per_dev = wkly_per_dev.agg(
                {'col_weekly_per_device': list(num_smry.values())}
            )
            smry_data: pd.DataFrame = smry_df.join(wkly_per_dev)
            smry_data.columns.set_names(
                names = ['variable', 'metric'], inplace = True
            )
            smry_data = smry_data.stack([0, 1]).rename('value')
            smry_data = smry_data.reset_index()

        else:
            for name, metr in num_smry.items():
                smry_vec.loc[
                    ('col_weekly_per_device', name)
                ] = metr(wkly_per_dev)
            smry_data = smry_vec.to_frame('value').reset_index(
                names = ['variable', 'metric'], inplace = False
            )

        return smry_data


    def plasma_summary(self,
                       strata: Optional[list[str]] = None,
                       pop: str = 'itt',
                       raw: bool = False) -> pd.DataFrame:
        """Calculate summary for numeric-type metrics, associated to plasma collection.

        Args:
            start (Optional[Union[dt.date, str]], optional): Lower bound. Defaults to None.
            end (Optional[Union[dt.date, str]], optional): Upper bound. Defaults to None.
            strata (Optional[list[str]], optional): Variables to split by. Defaults to None.
            pop (str, optional): Population. Defaults to 'itt'.
            raw (bool, optional): Include raw data. Defaults to False.

        Returns:
            pd.DataFrame: Summary data in the longer format.
        """

        bad_metr = {name: PlasmaCollection.__metrics[name]
                    for name in ['quantile_025', 'quantile_975']}

        num_smry = PlasmaCollection.__metrics.copy()

        if raw:
            bad_metr['raw_data'] = raw_data
            num_smry = num_smry | {'raw_data': raw_data}

        # plasma_metr = [metr for name, metr in PlasmaCollection.__metrics.items()
        #                if not (strata is None and name in bad_metr.keys())]

        smry = dict.fromkeys(
            ['target_vol', 'actual_vol',
             'yield', 'yield_resid',
             'duration_minutes', 'speed'],
            [metr for name, metr in num_smry.items()
             if not (strata is None and name in bad_metr.keys())]
        )

        data: pd.DataFrame = self.filter(query = {'POP': pop})

        if strata is not None:
            data = data.groupby(strata)

        smry_data: pd.DataFrame = data.agg(smry)

        if strata is not None:

            smry_data.columns.set_names(
                names = ['variable', 'metric'], inplace = True
            )
            smry_data = smry_data.stack([0, 1]).rename('value')
            smry_data = smry_data.reset_index()

        else:
            smry_vec: pd.Series = smry_data.unstack().dropna()

            for name, metr in bad_metr.items():
                for var in smry:
                    smry_vec.loc[(var, name)] = metr(data[var])

            for sev in self.surr_lvls['SEV']:
                smry_vec.loc[(sev, 'flags')] = raw_data(data[sev])

            smry_data = smry_vec.to_frame('value').reset_index(
                names = ['variable', 'metric'], inplace = False
            )

        return smry_data
