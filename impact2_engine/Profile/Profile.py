"""Donor profile monitoring module for IMPACT study"""

# %% Import libraries

__all__ = ['Safety']

import datetime as dt
from typing import Optional, Union, Any
# from numpy.typing import ArrayLike
from warnings import warn
import numpy as np
import pandas as pd
from impact2_engine.utils.utils import diff


# %% Safety class definition and methods

class Profile:
    """Safety monitoring module
    """

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

        self.__arrange()
        self.__enrich_levels()


    def __check_contents(self) -> None:

        try:
            with open(self.data_path, 'r', encoding = 'utf-8') as csv:
                header = csv.readline().rstrip().split(',')
        except ValueError:
            print("""Ensure path is correct and contains 'utf-8' csv-file,
                  with ',' delimeter and header on the 1st line.""")

        # head_up = [col.upper() for col in header]
        # cols_up = [col['name'].upper() for col in flatten(list(self.contents.values()))]
        names = [col['var'] for cols in self.contents.values() for col in cols]
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
        found_aes = [col['aes'] for col in self.contents['SEV']
                     if col['var'] == 'all_ae'][0]
        bool_flags = key_names['POP'] + key_names['SEV'] + found_aes
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            dict.fromkeys(key_names['DEM'] + key_names['VOL'], 'float') |
            dict.fromkeys(key_names['IDS'] + key_names['DTS'], 'object') |
            dict.fromkeys(bool_flags, 'bool')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names + found_aes,
            dtype = col_types,
            na_values = {col['var']: col['na']
                         for col in self.contents['VOL']
                         if 'na' in col},
            na_filter = na_filter  # detect missing value markers if True
        )

        # self.data.replace(
        #     to_replace = dict.fromkeys(
        #         bool_flags,
        #         {'True': True, 'False': False, np.nan: False}
        #     ),
        #     inplace = True
        # )

        for col in self.contents['DTS']:  # faster than parse_dates
            self.data[col['var']] = pd.to_datetime(
                self.data[col['var']], format = col['format']
            )

        # name_to_var = {col['name']: col['var']
        #                       for cols in self.contents.values()
        #                       for col in cols}
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


    def __enrich_levels(self) -> None:

        for col in self.contents['CAT']:
            col['lvl'] = list(self.data[col['var']].dtype.categories)

        self.strat_lvls = {col['var']: col['lvl']
                           for key, cols in self.contents.items()
                           for col in cols
                           if key == 'CAT'}

        self.surr_lvls = {key: [col['var'] for col in cols]
                          for key, cols in self.contents.items()
                          if key in ['POP', 'SEV']}

        self.comb_lvls = self.strat_lvls | self.surr_lvls


    def __surrogate(self, qry: dict[str, str]) -> dict[str, Any]:

        surr_qry: dict[str, Any] = qry.copy()

        for var, lvl in qry.items():
            if (var in self.surr_lvls.keys() and
                    lvl in self.surr_lvls[var]):
                surr_qry[lvl] = True
                del surr_qry[var]

        return surr_qry


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


    def summary_dem(self,
                    don_ids: list[str] | None = None) -> pd.DataFrame:
        """Summary of donor(s) demographics.

        Args:
            don_ids (list[str] | None, optional): Subset of donors. Defaults to None.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        if don_ids is not None:
            don_data = self.data.query(f"don_id in {str(don_ids)}")
        else:
            don_data = self.data

        dem = {'don_id':'donor_id',
               'site': 'site',
               'group': 'group',
               'status': 'status',
               'gender': 'gender',
               'age': 'age',
               'weight': 'first_weight',
               'bmi': 'first_bmi',
               'hct': 'first_hct'}

        smry_dem = don_data[dem.keys()].groupby('don_id').nth(0).reset_index()
        smry_dem.columns = dem.values()

        return smry_dem


    def summary_col(self,
                    don_ids: list[str] | None = None) -> pd.DataFrame:
        """Summary of donor(s) collection details.

        Args:
            don_ids (list[str] | None, optional): Subset of donors. Defaults to None.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        if don_ids is not None:
            don_data = self.data.query(f"don_id in {str(don_ids)}")
        else:
            don_data = self.data

        smry = {
            'col_id': [pd.Series.nunique],
            'sig_hyp': [np.sum],
            'all_ae': [np.sum],
            'duration_minutes': [np.mean],
            'speed': [np.mean],
            'weight': [diff],
            'bmi': [diff],
            'hct': [diff]
        }

        smry_tbl: pd.DataFrame = pd.pivot_table(
            data = don_data,
            index = 'don_id',
            values = list(smry.keys()),
            aggfunc = smry
        )

        return smry_tbl.swaplevel(axis = 1).sort_index(axis = 1)


    def chronology(self,
                   don_ids: list[str] | None = None) -> pd.DataFrame:
        """Donor(s) complete collection chronology.

        Args:
            don_ids (list[str] | None, optional): Subset of donors. Defaults to None.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        if don_ids is not None:
            don_data = self.data.query(f"don_id in {str(don_ids)}")
        else:
            don_data = self.data

        info = ['don_id', 'col_id', 'dev_id',
                'col_date', 'proc_start', 'proc_end',
                'duration_minutes', 'speed',
                'yield', 'target_vol', 'actual_vol',
                'weight', 'bmi', 'hct']

        chron: pd.DataFrame = don_data.reset_index()[info]
        chron['col_date_0'] = chron.groupby('don_id')['col_date'].transform('first')
        chron['days_total'] = (chron['col_date'] - chron['col_date_0']) // \
            pd.Timedelta(days = 1)
        chron['days_lag'] = chron.groupby('don_id')['days_total'].diff().replace(
            {np.nan: 0}
        ).astype(int)

        del chron['col_date_0']

        aes = [col['aes'] for col in self.contents['SEV']
               if col['var'] == 'all_ae'][0]

        chron['AE'] = don_data.reset_index()[aes].apply(
            lambda x: ', '.join(x[x].index.to_list()),
            axis = 1
        )

        return chron
