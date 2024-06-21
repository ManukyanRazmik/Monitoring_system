"""Conditional power monitoring module for IMPACT study"""

# pylint: disable=invalid-name

# %% Import libraries

__all__ = ['Power']

import datetime as dt
from warnings import warn
from typing import Optional, Union, Any
import math
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from scipy.stats import betabinom
from impact2_engine.utils.CalcCI import BinomData, RateDiff, Rate
# from impact2_engine.utils.utils import diff


# %% Safety class definition and methods

class Power:
    """Safety monitoring module
    """

    def __init__(self,
                 data_path: str,  # csv(','), utf-8, 1st line header
                 contents: dict[str, Any],
                 na_filter: bool = True):

        self.data_path = data_path
        self.contents = contents
        self.na_filter = na_filter
        self.plan = [col['plan'] for col in self.contents['IDS']
                     if col['var'] == 'col_id'][0]
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
        bool_flags = key_names['SEV']
        col_types = (
            dict.fromkeys(key_names['CAT'], 'category') |
            # dict.fromkeys(key_names['DEM'] + key_names['VOL'], 'float') |
            dict.fromkeys(key_names['IDS'] + key_names['DTS'], 'object') |
            dict.fromkeys(bool_flags, 'bool')
        )

        self.data = pd.read_csv(
            filepath_or_buffer = self.data_path,
            usecols = col_names,
            dtype = col_types,
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

        if (start is not None and start < self.data.index[0]):
            start = self.data.index[0]

        if (end is not None and end > self.data.index[-1]):
            end = self.data.index[-1]

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


    def posterior(self,
                  prior_p: ArrayLike = (.5, .5),
                  prior_n: ArrayLike = (1, 1),
                  severity: str = 'sig_hyp',
                  time_point: Union[dt.date, str] | None = None) -> pd.DataFrame:
        """Beta posterior distribution for binomial conjugate model.

        Args:
            prior_p (ArrayLike, optional): AE proportions. Defaults to (.5, .5).
            prior_n (ArrayLike, optional): Concentrations. Defaults to (1, 1).
            severity (str, optional): Severity group of AE. Defaults to 'sig_hyp'.
            time_point (Union[dt.date, str] | None, optional): Current date. Defaults to None.

        Returns:
            pd.DataFrame: Current data, augmented by prior and posterior.
        """

        mus: pd.Series = pd.Series(prior_p)
        kappas: pd.Series = pd.Series(prior_n)

        if mus.size != 2:
            raise ValueError(
                "Provide expected probabilities for 2 groups: A, B."
            )

        if kappas.size != 2:
            raise ValueError(
                """Provide confidence in respective group probabilities
                as 2 imaginary sample size equivalents."""
            )

        if not all(mus.between(0, 1)):
            raise ValueError(
                "Group probabilities should be within [0, 1]."
            )

        grp: pd.DataFrame = self.filter(end = time_point).groupby('group')
        data = grp.agg(
            n_aes = (severity, np.sum),
            n_col = ('col_id', pd.Series.nunique)
        ).reset_index()

        data['observed_p'] = data['n_aes'] / data['n_col']
        data['expected_p'] = mus
        data['prior_n'] = kappas
        data['post_n'] = kappas + data['n_col']
        data['weighted_p'] = (mus * kappas + data['n_aes']) / data['post_n']
        data['success'] = data['weighted_p'] * data['post_n']
        data['failure'] = (1 - data['weighted_p']) * data['post_n']
        data['margin'] = [col['margin'] for col in self.contents['SEV']
                          if col['var'] == severity][0] / 100

        return data


    def simulate(self,
                 samples: int = 50000,
                 step: int = 1000,
                 alpha: float = .05,
                 method: str = 'wald_cc',
                 **kwargs: Any) -> pd.DataFrame:
        """Use posterior to simulate the AE for range of sample sizes.

        Args:
            samples (int, optional): Number of simulations. Defaults to 10000.
            step (int, optional): Step for number of trials. Defaults to 1000.
            alpha (float, optional): Error 1 rate for NI test. Defaults to .05.
            method (str, optional): CI method for NI test. Defaults to 'wald_cc'.

        Returns:
            pd.DataFrame: Power calculations summary table.
        """

        post = self.posterior(**kwargs).set_index('group')

        trials = np.arange(
            start = math.ceil(max(post['n_col']) / step) * step,
            stop = math.ceil(self.plan / 2 / step) * step + 1,
            step = step,
            dtype = int
        )

        binom_data: list[BinomData] = []

        for group in ['A', 'B']:

            bb_rvs = betabinom.rvs(
                n = (trials - post.loc[group, 'n_col'])[:, None],
                a = post.loc[group, 'success'],
                b = post.loc[group, 'failure'],
                size = (trials.size, samples)
            )
            binom_data.append(
                BinomData(
                    x = np.reshape(bb_rvs, trials.size * samples) + \
                        post.loc[group, 'n_aes'],
                    n = np.repeat(trials, samples) + \
                        post.loc[group, 'n_col']
                )
            )

        rate_diff = RateDiff(
            data = (binom_data[0], binom_data[1]),
            limits = 'both',
            conf = 1 - 2 * alpha
        )

        lim = np.reshape(
            rate_diff.calc_ci(method = method).to_numpy().T,
            newshape = (2, trials.size, samples)
        )

        smry = pd.DataFrame(
            data = {
                'width_mean': np.mean(lim[1] - lim[0], 1),
                'width_power': np.mean(
                    lim[1] - lim[0] < 1 * post['margin'][0], axis = 1
                ),
                'u_mean': np.mean(lim[1], 1),
                'u_std': np.std(lim[1], 1),
                'u_median': np.median(lim[1], 1),
                'u_025': np.quantile(lim[1], .025, 1),
                'u_975': np.quantile(lim[1], .975, 1),
                'test_power': np.mean(lim[1] < post['margin'][0], 1)
            },
            index = trials
        )

        pwr = Rate(
            data = BinomData(
                x = smry['test_power'] * samples,
                n = pd.Series(
                    np.repeat(samples, trials.size),
                    index = trials
                )
            ),
            conf = .99
        )
        pwr_ci = pwr.calc_ci('clopper_pearson').add_prefix('test_pwr_')

        return pd.concat([smry.round(6), pwr_ci.round(4)], axis = 1) * 100
