"""Safety monitoring module for IMPACT study"""

# %% Import libraries

__all__ = ['Safety']

import datetime as dt
from typing import Optional, Union, Any
# from numpy.typing import ArrayLike
from warnings import warn
import numpy as np
import pandas as pd
import sidetable
from impact2_engine.utils.CalcCI import BinomData, Rate, RateDiff


# %% Safety class definition and methods

class Safety:
    """Safety monitoring module
    """

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.data = pd.DataFrame()
        self.missing: Optional[pd.DataFrame] = None
        self.update_data()


    def update_data(self) -> None:
        """Check, load and preprocess data at request"""

        self.__check_contents()
        self.__load_data(na_filter = self.na_filter)

        if self.na_filter:
            self.__handle_na()

        # self.__discretize_demo()
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
            # dict.fromkeys(key_names['DEM'], 'float') |
            dict.fromkeys(key_names['IDS'], 'object') |
            dict.fromkeys(bool_flags, 'bool')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names + found_aes,
            dtype = col_types,
            parse_dates = key_names['DAT'],
            na_filter = na_filter  # detect missing value markers if True
        )

        # self.data.replace(
        #     to_replace = dict.fromkeys(
        #         bool_flags,
        #         {'True': True, 'False': False, np.nan: False}
        #     ),
        #     inplace = True
        # )

        # replace columns by standardized variable names ?

        # name_to_var = {col['name']: col['var']
        #                       for cols in self.contents.values()
        #                       for col in cols}
        # self.data.columns = [name_to_var.get(item, item)
        #                      for item in self.data.columns]

        dates = self.contents['DAT'][0]['var']
        self.data.set_index(dates, inplace = True)
        self.data.sort_index(inplace = True)
        # self.data.reset_index(inplace = True)


    def __handle_na(self) -> None:

        if self.data.isnull().values.any():
            warn(
                """Found missing values. Keep only compelete cases.
                Inspect 'missing' field to find the source of NA.
                Disable na_filter to read the original data."""
            )
            self.missing = self.data[self.data.isna().any(axis = 1)]
            self.data.dropna(axis = 0, how = 'any', inplace = True)


    # def __discretize_demo(self) -> None:

    #     for col in self.contents['DEM']:

    #         if (isinstance(col['bin'], list) and
    #             isinstance(col['lvl'], list) and
    #             len(col['bin']) == len(col['lvl']) + 1 and
    #             len(col['lvl']) > 1):

    #             self.data[col['var']] = pd.cut(
    #                 self.data[col['var']],
    #                 bins = col['bin'],
    #                 labels = col['lvl'],
    #                 right = False
    #             )

    #         elif (isinstance(col['bin'], int) and
    #               isinstance(col['lvl'], list) and
    #               len(col['lvl']) == col['bin']):

    #             self.data[col['var']] = pd.qcut(
    #                 self.data[col['var']],
    #                 q = col['bin'],
    #                 labels = col['lvl']
    #             )

    #         else:
    #             raise ValueError(
    #                 """Numerical columns in contents of 'DEM' should contain:
    #                 the 'lvl' list of labels and 'bin' into dicrete categories,
    #                 provided either as cutpoints (list) or number of bins (int)."""
    #             )


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


    def summary_donat(self) -> pd.DataFrame:
        """Donation/donor summary information, split by 'group'.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        plan = [col['plan'] for col in self.contents['IDS']
                if 'plan' in col][0]

        smry = dict.fromkeys(['don_id', 'col_id'], [pd.Series.nunique])

        smry_tbl: pd.DataFrame = pd.pivot_table(
            data = self.data,
            index = 'group',
            values = list(smry.keys()),
            aggfunc = smry
        )
        smry_tbl = smry_tbl.stb.subtotal()

        smry_tbl[('col_id', 'pct')] = smry_tbl[('col_id', 'nunique')] / \
            smry_tbl.loc['grand_total', ('col_id', 'nunique')] * 100

        smry_tbl.loc['grand_total', ('col_id', 'pct')] = \
            smry_tbl.loc['grand_total', ('col_id', 'nunique')] / plan * 100

        per_don = self.data.groupby(['group', 'don_id'])
        per_don = per_don['col_id'].nunique().groupby('group')
        per_don = per_don.mean().rename(('col_per_don', 'mean'))
        smry_tbl = smry_tbl.join(per_don)

        smry_tbl.loc['grand_total', ('col_per_don', 'mean')] = (
            smry_tbl.loc['A', ('col_per_don', 'mean')] * \
                smry_tbl.loc['A', ('col_id', 'nunique')] +
            smry_tbl.loc['B', ('col_per_don', 'mean')] * \
                smry_tbl.loc['B', ('col_id', 'nunique')]
        ) / smry_tbl.loc['grand_total', ('col_id', 'nunique')]

        smry_tbl.columns.set_names(['variable', 'metric'], inplace = True)
        smry_tbl.index.set_names('group', inplace = True)

        return smry_tbl


    def summary_risk(self) -> pd.DataFrame:
        """AE severity rates summary, split by 'group'.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        sev_lst = [col['var'] for col in self.contents['SEV']]

        smry = (
             dict.fromkeys(sev_lst, [np.sum]) |
             {'col_id': [pd.Series.nunique]}
        )

        smry_tbl: pd.DataFrame = pd.pivot_table(
            data = self.data,
            index = 'group',
            values = list(smry.keys()),
            aggfunc = smry
        )
        smry_tbl = smry_tbl.stb.subtotal()

        for sev in sev_lst:
            smry_tbl[(sev, 'pct')] = smry_tbl[(sev, 'sum')] / \
                smry_tbl[('col_id', 'nunique')] * 100

        smry_tbl.drop(columns = ('col_id', 'nunique'), inplace = True)
        smry_tbl.columns.set_names(['variable', 'metric'], inplace = True)
        smry_tbl.index.set_names('group', inplace = True)

        return smry_tbl


    def summary_aes(self,
                    sev: str,
                    pop: str = 'itt',
                    conditional: bool = False) -> pd.DataFrame:
        """Summary of AE incidences by type, split by 'group'.

        Args:
            sev (str): Severity group of AE.
            pop (str, optional): Population. Defaults to 'itt'.
            conditional (bool, optional): Within severity group. Defaults to False.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        if sev not in self.surr_lvls['SEV']:
            raise ValueError(
                f"Severity group should be one of {self.surr_lvls['SEV']}"
            )

        if pop not in self.surr_lvls['POP']:
            raise ValueError(
                f"Population should be one of {self.surr_lvls['POP']}"
            )

        aes_lst = [col['aes'] for col in self.contents['SEV']
                   if col['var'] == sev][0]

        data = self.filter(query = {'POP': pop})

        smry = (
            dict.fromkeys([sev] + aes_lst, [np.sum]) |
            {'col_id': [pd.Series.nunique]}
        )

        smry_tbl: pd.DataFrame = pd.pivot_table(
            data = data,
            index = 'group',
            values = list(smry.keys()),
            aggfunc = smry
        )

        smry_tbl = smry_tbl.stb.subtotal()

        # smry_tbl[(sev, 'pct_col')] = smry_tbl[(sev, 'sum')] / \
        #     smry_tbl[('col_id', 'nunique')] * 100

        smry_tbl.rename(columns = {'nunique': 'sum'}, level = 1, inplace = True)

        denominator = (sev, 'sum') if conditional else ('col_id', 'sum')

        for event in list(smry.keys()):

            smry_tbl[(event, 'pct')] = smry_tbl[(event, 'sum')] / \
                smry_tbl[denominator] * 100

            # smry_tbl.replace(
            #     to_replace = {(event, 'pct'): {np.nan: None}},
            #     inplace = True
            # )

        smry_tbl[('col_id', 'pct')] = np.nan if conditional else \
            smry_tbl[('col_id', 'sum')] / \
            smry_tbl.loc['grand_total', ('col_id', 'sum')] * 100

        smry_tbl.columns.set_names(['variable', 'metric'], inplace = True)
        smry_tbl.index.set_names('group', inplace = True)

        return smry_tbl


    def summary_sev(self,
                    strata: list[str],
                    sev: str,
                    pop: str = 'itt') -> pd.DataFrame:
        """Summary of AE incidences by severity, split by user-defined 'strata'.

        Args:
            strata (list[str]): Categorical types to split by.
            sev (str): Severity group of interest.
            pop (str, optional): Population. Defaults to 'itt'.

        Returns:
            pd.DataFrame: Summary table in the wide format.
        """

        if sev not in self.surr_lvls['SEV']:
            raise ValueError(
                f"Severity group should be one of {str(self.surr_lvls['SEV'])}"
            )

        if pop not in self.surr_lvls['POP']:
            raise ValueError(
                f"Population should be one of {str(self.surr_lvls['POP'])}"
            )

        if not (isinstance(strata, list) and len(strata) > 0):
            raise TypeError(
                "'strata' should be non-empty list[str] of categorical types."
            )

        if bool(np.setdiff1d(strata, list(self.strat_lvls.keys()))):
            raise ValueError(
                f"""'strata' should be one of {str(list(self.strat_lvls.keys()))}.
                % normalization for the innermost (last) stratification."""
            )

        data = self.filter(query = {'POP': pop})

        smry_tbl: pd.DataFrame = pd.pivot_table(
            data = data,
            index = strata,
            values = [sev],
            aggfunc = {sev: [np.sum]}
        )

        if len(strata) == 1:
            pct = smry_tbl
        else:
            pct = smry_tbl.groupby(strata[:-1])

        smry_tbl[(sev, 'pct')] = \
            pct[[(sev, 'sum')]].transform(lambda x: x/x.sum() * 100)

        smry_tbl = smry_tbl.stb.subtotal().replace(
            to_replace = {'pct': {np.nan: None}}
        )

        smry_tbl.columns.set_names(['variable', 'metric'], inplace = True)
        smry_tbl.index.set_names(strata, inplace = True)

        return smry_tbl


    def calc_ci(self,
                sev: str,
                pop: str = 'itt',
                limits: str = 'both',
                conf: float = .89) -> pd.DataFrame:
        """Confidence interval computation.

        Args:
            sev (str): Severity group of AE.
            limits (str, optional): 'upper'/'lower'. Defaults to 'both'.
            conf (float, optional): Nominal confidence level. Defaults to .89.

        Returns:
            pd.DataFrame: Results for CI by different methods + margin.
        """

        if sev not in self.surr_lvls['SEV']:
            raise ValueError(
                f"Severity group should be one of {self.surr_lvls['SEV']}"
            )

        if pop not in self.surr_lvls['POP']:
            raise ValueError(
                f"Population should be one of {self.surr_lvls['POP']}"
            )

        data: pd.DataFrame = self.filter(query = {'POP': pop})
        stats = data.groupby('group').agg(
            n = ('col_id', pd.Series.nunique),
            x = (sev, np.sum)
        )
        stats['p'] = stats['x'] / stats['n']
        binom_data = (
            BinomData(x = stats.loc['A', 'x'], n = stats.loc['A', 'n']),
            BinomData(x = stats.loc['B', 'x'], n = stats.loc['B', 'n'])
        )
        rate_diff = RateDiff(
            data = binom_data,
            limits = limits,
            conf = conf
        )
        ci_list = []
        for method in RateDiff.METHODS:
            ci_list.append(
                rate_diff.calc_ci(method = method)
            )
        ci_smry = pd.concat(ci_list).transform(lambda x: x * 100)
        ci_smry.index = RateDiff.METHODS

        p_1 = stats.loc['B', 'p'] * 100
        p_0 = stats.loc['A', 'p'] * 100
        ci_smry['risk_diff'] = p_1 - p_0
        ci_smry['margin'] = [col['margin']
                             for col in self.contents['SEV']
                             if col['var'] == sev][0]

        return ci_smry


    def summary_longitudinal(self,
                             method: str,
                             sev: str,
                             pop: str = 'itt',
                             aggregate: str = 'w',
                             conf: float = .89,
                             start: Optional[Union[dt.date, str]] = None,
                             end: Optional[Union[dt.date, str]] = None) -> pd.DataFrame:
        """Calculate AE rates CIs for aggregated data, per group and pulled.

        Args:
            method (str): One of available methods.
            sev (str): Severity group of AE.
            pop (str, optional): Population. Defaults to 'itt'.
            aggregate (str, optional): Period of observations. Defaults to week ('w').
            start (Optional[Union[dt.date, str]], optional): Lower bound. Defaults to None.
            end (Optional[Union[dt.date, str]], optional): Upper bound. Defaults to None.

        Returns:
            pd.DataFrame: Full CI summary, split by timepoints.
        """

        if aggregate not in {'d', 'w', 'm'}:
            raise ValueError(
                "'aggregate' observations over one of ['d', 'w', 'm']."
            )

        data = self.filter(start = start, end = end, query = {'POP': pop})

        period = pd.Series(data.index).dt.to_period(aggregate)
        rate_data = data.assign(
            **{aggregate: period.dt.start_time.values}
        ).groupby(['group', aggregate]).agg(
            n = ('col_id', pd.Series.nunique),
            x = (sev, np.sum)
        ).unstack(0)

        rate_data.loc[:, ('n', 'pulled')] = \
            rate_data.loc[:, ('n', 'A')] + rate_data.loc[:, ('n', 'B')]
        rate_data.loc[:, ('x', 'pulled')] = \
            rate_data.loc[:, ('x', 'A')] + rate_data.loc[:, ('x', 'B')]

        for sample in ['A', 'B', 'pulled']:

            sample_data = BinomData(
                x = rate_data.loc[:, ('x', sample)],
                n = rate_data.loc[:, ('n', sample)]
            )
            sample_ci = Rate(sample_data, conf = conf).calc_ci(method)
            sample_ci.columns = pd.MultiIndex.from_product(
                [sample_ci.columns,
                 [sample]]
            )
            sample_ci.index = rate_data.index
            rate_data = rate_data.join(sample_ci)
            rate_data.loc[:, ('rate', sample)] = \
                rate_data.loc[:, ('x', sample)] / rate_data.loc[:, ('n', sample)]

        rate_data = rate_data.swaplevel(axis = 1).sort_index(axis = 1)
        rate_data.columns.set_names(['sample', 'metric'], inplace = True)
        rate_data.index.set_names('period', inplace = True)

        return rate_data
