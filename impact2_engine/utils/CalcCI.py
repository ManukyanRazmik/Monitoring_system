"""Confidence interval calculation
for the difference of binomial proportions
"""

# pylint: disable=invalid-name

from typing import NamedTuple, Any, Callable # Self
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from scipy.stats import norm, beta, t
from scipy.optimize import minimize


class BinomData(NamedTuple):
    """Aggregated binomial data from Bernoulli trials.

    Args:
        NamedTuple (int): # counts/trials for independent samples.
    """

    x: ArrayLike
    n: ArrayLike


class Rate:
    """CI calculation (vectorized) for a single proportion/rate."""
    METHODS: list[str] = [
        'agresti_coull',
        'clopper_pearson',
        'clopper_pearson_cc',
        'jeffreys',
        'uniform',
        'wald',
        'wald_cc',
        'wilson',
        'wilson_cc'
    ]

    __LIMITS: set[str] = {'lower', 'upper', 'both'}

    def __init__(self,
                 data: BinomData,
                 limits: str = 'both',
                 conf: float = .89) -> None:

        if not isinstance(data, BinomData):
            raise ValueError(
                """'data' should be aggregated counts BinomData(x, n)."""
            )

        if limits not in Rate.__LIMITS:
            raise ValueError(
                f"'limits' should be one of {list(Rate.__LIMITS)}."
            )

        if not 0 <= conf <= 1:
            raise ValueError(
                "The nominal 'conf' level should be within [0, 1]."
            )

        self.x_vec: pd.Series = pd.Series(data.x)
        self.n_vec: pd.Series = pd.Series(data.n)

        self.limits: str = limits
        self.conf: float = conf

        if limits in {'lower', 'upper'}:
            self.z_conf: float = norm.ppf(conf)
        else:
            self.z_conf = norm.ppf((1 + conf) / 2)

    def asymp_norm(self,
                   rate: pd.Series,
                   var: pd.Series,
                   cont_corr: bool = False) -> pd.DataFrame:
        """Normal approximation with arbitrary centering and variances.

        Args:
            rate (float): Rate center (point estimate).
            var (float): Rate variance estimate.

        Returns:
            pd.DataFrame: CI boundaries, truncated to [0, 1].
        """

        err: pd.Series = np.sqrt(var)

        low: pd.Series = rate - self.z_conf * err

        if cont_corr:
            c_c = 1 / 2 / self.n_vec
            low = low - c_c

        low.clip(lower = 0, inplace = True)

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp: pd.Series = rate + self.z_conf * err

        if cont_corr:
            c_c = 1 / 2 / self.n_vec
            upp = upp + c_c

        upp.clip(upper = 1, inplace = True)

        if self.limits == 'upper':
            return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def wald(self,
             cont_corr: bool = False) -> pd.DataFrame:
        """Naive (wrong) confidence interval.

        Args:
            cont_corr (bool, optional): Continuity correction. Defaults to False.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        rate = self.x_vec / self.n_vec
        var = rate * (1 - rate) / self.n_vec

        return self.asymp_norm(rate, var, cont_corr)

    def wald_cc(self) -> pd.DataFrame:
        """Wald interval with Yates continuity correction.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.wald(cont_corr = True)

    def agresti_coull(self) -> pd.DataFrame:
        """Agresti-Coull's pseudo-counts recentered CI.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        x_vec = self.x_vec + self.z_conf ** 2 / 2
        n_vec = self.n_vec + self.z_conf ** 2
        rate = x_vec / n_vec
        var = rate * (1 - rate) / n_vec

        return self.asymp_norm(rate, var)

    def wilson(self,
               cont_corr: bool = False) -> pd.DataFrame:
        """Wilson score interval from chi-sq test inversion."""

        n_vec = self.n_vec
        w_2 = self.z_conf ** 2 / n_vec

        if self.limits in {'lower', 'both'}:

            p_corr = self.x_vec / n_vec

            if cont_corr:
                p_corr = p_corr - 1 / n_vec
                p_corr.clip(lower = 0, inplace = True)

            rate = p_corr + w_2 / 2
            var = (p_corr * (1 - p_corr) + w_2 / 4) / n_vec

            low = self.asymp_norm(rate, var)['LL'] / (1 + w_2)

            if self.limits == 'lower':
                return low.to_frame('LL')

        if self.limits in {'upper', 'both'}:

            p_corr = self.x_vec / n_vec

            if cont_corr:
                p_corr = p_corr + 1 / n_vec
                p_corr.clip(upper = 1, inplace = True)

            rate = p_corr + w_2 / 2
            var = (p_corr * (1 - p_corr) + w_2 / 4) / n_vec

            upp = self.asymp_norm(rate, var)['UL'] / (1 + w_2)

            if self.limits == 'upper':
                return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def wilson_cc(self) -> pd.DataFrame:
        """Wilson score interval with continuity correction.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.wilson(cont_corr = True)

    def clopper_pearson(self,
                        cont_corr: bool = False) -> pd.DataFrame:
        """The 'exact' inversion of the binomial test.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        a_vec = self.x_vec
        b_vec = self.n_vec - self.x_vec
        b_vec.index = a_vec.index

        low = pd.Series(
            beta.ppf(1 - self.conf, a_vec, b_vec + 1),
            index = a_vec.index
        )
        low[a_vec == 0] = 0

        if cont_corr:
            low_cc = pd.Series(
                beta.ppf(1 - self.conf, a_vec + 1, b_vec),
                index = a_vec.index
            )
            low_cc[b_vec == 0] = 1
            low = (low + low_cc) / 2

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp = pd.Series(
            beta.ppf(self.conf, a_vec + 1, b_vec),
            index = a_vec.index
        )
        upp[b_vec == 0] = 1

        if cont_corr:
            upp_cc = pd.Series(
                beta.ppf(self.conf, a_vec, b_vec + 1),
                index = a_vec.index
            )
            upp_cc[a_vec == 0] = 0
            upp = (upp + upp_cc) / 2

        if self.limits == 'upper':
            return upp.to_frame('UL')

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def clopper_pearson_cc(self) -> pd.DataFrame:
        """'Exact' CI with continuity correction.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.clopper_pearson(cont_corr = True)

    def bayes_hdi(self,
                  kappa: float = 2) -> pd.DataFrame:
        """Bayesian unimodal 2-sided Highest Density Interval.

        Args:
            kappa (float): Concentration parameter = alpha + beta.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        if self.limits != 'both':
            raise KeyError(
                "HDI must have 'both' limits."
            )

        a_vec = self.x_vec + kappa / 2
        b_vec = self.n_vec - self.x_vec + kappa / 2

        def post(x: ArrayLike) -> Any:
            """Posterior inverse cumulative probability distribution.

            Args:
                x (ArrayLike): Probability.

            Returns:
                Any: Quantile value.
            """

            return beta.ppf(x, a_vec, b_vec)

        def ci_width_2(low_tail: np.ndarray[Any, np.dtype[Any]],
                       cred_mass: np.ndarray[Any, np.dtype[Any]],
                       ppf: Callable[..., np.ndarray[Any, np.dtype[Any]]],
                       **kwargs: Any) -> float:
            """Positive optimization function.

            Args:
                low_tail (np.ndarray[Any]): Lower CI boundary.
                cred_mass (np.ndarray[Any]): Credible probability mass.
                ppf (Callable[..., np.ndarray[Any, pd.dtype[Any]]]): \
                    Posterior quantile function.

            Returns:
                float: CI width sum of squares.
            """

            width_2: float = np.sum(
                (
                    ppf(cred_mass + low_tail, **kwargs) - \
                        ppf(low_tail, **kwargs)
                ) ** 2
            )

            return width_2

        cred_mass = np.resize([self.conf], a_vec.size)
        init_low = np.resize([(1 - self.conf) / 2], a_vec.size)

        out = minimize(
            fun = ci_width_2,
            x0 = init_low,
            args = (cred_mass, post),
            tol = 1e-8
            # bounds =
        )

        low = pd.Series(post(out.x)).set_axis(a_vec)
        upp = pd.Series(post(out.x + cred_mass)).set_axis(a_vec)

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)


    def bayes(self,
              kappa: float = 1) -> pd.DataFrame:
        """Bayesian equal-tailed CI, with symmetric beta prior.

        Args:
            kappa (float): Concentration parameter = alpha + beta.

        Returns:
            pd.DataFrame: CI boudaries.
        """

        a_vec = self.x_vec + kappa / 2
        b_vec = self.n_vec - self.x_vec + kappa / 2

        low = pd.Series(beta.ppf(1 - self.conf, a_vec, b_vec))
        low.index = a_vec.index
        low[a_vec == 0] = 0

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp = pd.Series(beta.ppf(self.conf, a_vec, b_vec))
        upp.index = b_vec.index
        upp[b_vec == 0] = 1

        if self.limits == 'upper':
            return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def jeffreys(self) -> pd.DataFrame:
        """Jeffreys prior Bayesian confidence interval.\
            Approximates Mid-P corrected Clopper-Pearson.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.bayes(kappa = 1)

    def uniform(self) -> pd.DataFrame:
        """Uniform prior Bayesian confidence interval.

        Returns:
            pd.DataFrame: CI boundaries.
        """
        return self.bayes(kappa = 2)

    def calc_ci(self, method: str, **params: Any) -> pd.DataFrame:
        """Calculate CI by a particular method.

        Args:
            method (str): One of available methods.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        if method not in Rate.__dict__:
            raise ValueError(
                f"Choose 'method' from {list(Rate.__dict__.keys())}"
            )

        return Rate.__dict__[method](self, **params)

    def summary(self,
                **methods: dict[str, Any]) -> pd.DataFrame:
        """CI summary by various methods, with horizontal layout."""

        ci_list = []

        if not methods:
            methods = dict.fromkeys(Rate.METHODS, {})

        for method, params in methods.items():
            ci_meth = self.calc_ci(method, **params)  # pylint: disable=not-a-mapping
            ci_meth.columns = pd.MultiIndex.from_product(
                [[method], ci_meth.columns]
            )
            ci_list.append(ci_meth)

        ci_smry = pd.concat(ci_list, axis = 1)

        return ci_smry


class RateDiff:
    """CI calculation by various methods"""

    METHODS: list[str] = [
        'agresti_caffo',
        'anderson_hauck',
        'brown_li',
        'jeffreys',
        'jeffreys_hybrid',
        'jeffreys_perks',
        'jeffreys_perks_cc',
        'haldane',
        'haldane_cc',
        'newcombe',
        'newcombe_cc',
        'uniform',
        'uniform_hybrid',
        'wald',
        'wald_cc',
        'yule',
        'yule_mod'
    ]

    __LIMITS: set[str] = {'lower', 'upper', 'both'}

#     __subclasses: dict[str, Type[Self]] = {}

#     def __init_subclass__(cls, /, _INTERVAL_TYPE, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls._INTERVAL_TYPE = _INTERVAL_TYPE
#         cls.__subclasses[cls._INTERVAL_TYPE] = cls

#     @classmethod
#     def create(cls, interval_type, params):
#         if interval_type not in cls.__subclasses:
#             raise ValueError(f'Bad interval type {interval_type}')

#         return cls.__subclasses[interval_type](params)


# class RateDiff(CalcCI):
#     _INTERVAL_TYPE = 'rate_diff'

    def __init__(self,
                 data: tuple[BinomData, BinomData],
                 limits: str = 'both',
                 conf: float = .89) -> None:

        if not (isinstance(data[0], BinomData) and
                isinstance(data[1], BinomData)):
            raise ValueError(
                """'data' should be a pair tuple of BinomData(x, n)."""
            )

        if not len(data[0]) == len(data[1]):
            raise ValueError(
                "Sequences of counts should have the same length (vectorized)."
            )

        if limits not in RateDiff.__LIMITS:
            raise ValueError(
                f"'limits' should be one of {list(RateDiff.__LIMITS)}."
            )

        if not 0 <= conf <= 1:
            raise ValueError(
                "The nominal 'conf' level should be within [0, 1]."
            )

        self.x_1 = pd.Series(data[1].x)
        self.n_1 = pd.Series(data[1].n)
        self.x_0 = pd.Series(data[0].x)
        self.n_0 = pd.Series(data[0].n)

        # self.x_ = tuple(map(pd.Series, (data[0].x, data[1].x)))  # pylint: disable=invalid-name
        # self.n_ = tuple(map(pd.Series, (data[0].n, data[1].n)))  # pylint: disable=invalid-name
        # self.data = data

        self.limits: str = limits
        self.conf: float = conf

        if limits in {'lower', 'upper'}:
            self.z_conf: float = norm.ppf(conf)
        else:
            self.z_conf = norm.ppf((1 + conf) / 2)

    def asymp_norm(self,
                   p_1: pd.Series,
                   p_0: pd.Series,
                   var_1: pd.Series,
                   var_0: pd.Series,
                   cont_corr: str = '') -> pd.DataFrame:
        """Normal approximation with arbitrary centering and variances.

        Args:
            p_1 (float): 1st sample center.
            p_2 (float): 2nd sample center.
            var_1 (float): 1st sample variance.
            var_2 (float): 2nd sample variance.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        # theta = p_[1] - p_[0]
        # err = np.sqrt(var_[0] + var_[1])

        err = self.z_conf * np.sqrt(var_1 + var_0)

        if cont_corr == 'yates':
            c_c = (1 / self.n_1 + 1 / self.n_0) / 2
            err = err + c_c

        if cont_corr == 'anderson_hauck':
            n_df = pd.concat([self.n_1, self.n_0], axis = 1)
            c_c = .5 / n_df.min(axis = 1).set_axis(self.n_1.index)
            err = err + c_c

        low: pd.Series = p_1 - p_0 - err
        low.clip(lower = -1, inplace = True)

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp: pd.Series = p_1 - p_0 + err
        upp.clip(upper = 1, inplace = True)

        if self.limits == 'upper':
            return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def wald(self,
             cont_corr: str = '') -> pd.DataFrame:
        """Standard Wald confidence interval.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        # p_: Iterable[ArrayLike] = [sample.x / sample.n
        #                            for sample in self.data]
        # var_ = [p * (1 - p) / sample.n
        #         for p, sample in zip(p_, self.data)]

        p_1 = self.x_1 / self.n_1
        p_0 = self.x_0 / self.n_0
        var_1 = p_1 * (1 - p_1) / self.n_1
        var_0 = p_0 * (1 - p_0) / self.n_0

        out = self.asymp_norm(
            p_1, p_0, var_1, var_0, cont_corr
        )

        return out

    def wald_cc(self) -> pd.DataFrame:
        """Wald interval with continuity correction.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.wald(cont_corr = 'yates')

    def anderson_hauck(self) -> pd.DataFrame:
        """Anderson & Hauck (1986) using unbiased variances \
            and less conservative continuity correction\
                DOI:10.2307/2684618

        Returns:
            pd.DataFrame: CI boundaries.
        """

        p_1 = self.x_1 / self.n_1
        p_0 = self.x_0 / self.n_0
        var_1 = p_1 * (1 - p_1) / (self.n_1 - 1)
        var_0 = p_0 * (1 - p_0) / (self.n_0 - 1)

        out = self.asymp_norm(
            p_1, p_0, var_1, var_0,
            cont_corr = 'anderson_hauck'
        )

        return out


    def yule(self,
             modified: bool = False) -> pd.DataFrame:
        """Yule (1911)'s pulled CI, original and with modification.

        Args:
            modified (bool, optional): Reweighted estimate. Defaults to False.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        n_1 = self.n_1
        n_0 = self.n_0
        y_1 = self.x_1
        y_0 = self.x_0
        p_1 = y_1 / n_1
        p_0 = y_0 / n_0

        if modified:
            y_1 = n_0 * p_1
            y_0 = n_1 * p_0

        bar_p = (y_1 + y_0) / (n_1 + n_0)
        var_1 = bar_p * (1 - bar_p) / n_1
        var_0 = bar_p * (1 - bar_p) / n_0

        return self.asymp_norm(p_1, p_0, var_1, var_0)

    def yule_mod(self) -> pd.DataFrame:
        """Modified Yule's estimate, according to Brown & Li (2005).

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.yule(modified = True)

    def agresti_caffo(self,
                      pseudo: float = 4) -> pd.DataFrame:
        """Agresti-Caffo's pseudo-counts recentered CI.\
            DOI:10.2307/2685779

        Args:
            pseudo (float, optional): Total pseudo-counts. Defaults to 4.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        x_1 = self.x_1 + pseudo / 4
        x_0 = self.x_0 + pseudo / 4
        n_1 = self.n_1 + pseudo / 2
        n_0 = self.n_0 + pseudo / 2
        p_1 = x_1 / n_1
        p_0 = x_0 / n_0
        var_1 = p_1 * (1 - p_1) / n_1
        var_0 = p_0 * (1 - p_0) / n_0

        return self.asymp_norm(p_1, p_0, var_1, var_0)

    def pseudo_bayes(self,
                     kappa: float,
                     hybrid: bool) -> pd.DataFrame:
        """Bayesian estimate of independent proportions for various priors.

        Args:
            kappa (float): Concentration parameter = alpha + beta
            approx (bool): Binomial variance estimates, instead of beta.

        Returns:
            pd.DataFrame: CI coundaries.
        """

        p_1 = (self.x_1 + kappa / 2) / (self.n_1 + kappa)
        p_0 = (self.x_0 + kappa / 2) / (self.n_0 + kappa)

        if hybrid:
            var_1 = p_1 * (1 - p_1) / self.n_1
            var_0 = p_0 * (1 - p_0) / self.n_0

        else:
            var_1 = p_1 * (1 - p_1) / (self.n_1 + kappa + 1)
            var_0 = p_0 * (1 - p_0) / (self.n_0 + kappa + 1)

        return self.asymp_norm(p_1, p_0, var_1, var_0)


    def jeffreys(self,
                 hybrid: bool = False) -> pd.DataFrame:
        """Jeffrey's prior pseudo-Bayesian confidence interval.

        Args:
            approx (bool, optional): Binomial variance. Defaults to False.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.pseudo_bayes(kappa = 1, hybrid = hybrid)


    def jeffreys_hybrid(self) -> pd.DataFrame:
        """Jeffrey's prior, binomial variance for posterior proportion.

        Returns:
            pd.DataFrame: CI boundaries.
        """


        return self.jeffreys(hybrid = True)


    def uniform(self,
                hybrid: bool = False) -> pd.DataFrame:
        """Uniform prior (Berry?) pseudo-Bayesian confidence interval.

        Args:
            approx (bool, optional): Binomial variance. Defaults to False.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.pseudo_bayes(kappa = 2, hybrid = hybrid)


    def uniform_hybrid(self) -> pd.DataFrame:
        """Uniform prior, binomial variance for posterior proportion.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.uniform(hybrid = True)


    def beal(self,
             alpha: float,
             cont_corr: bool = False) -> pd.DataFrame:
        """Two methods from Beal's (1987).\
            DOI:10.2307/2531547
            DOI:10.1007/s44199-023-00054-8

        Args:
            method (string): Weighted proportion.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        n_ = (self.n_0, self.n_1)
        theta = self.x_1 / n_[1] - self.x_0 / n_[0]
        tau = (self.x_1 + alpha + 1) / (n_[1] + 2 * (alpha + 1)) + \
            (self.x_0 + alpha + 1) / (n_[0] + 2 * (alpha + 1))

        a = self.z_conf ** 2 * (1 / n_[1] + 1 / n_[0]) / 4
        b = self.z_conf ** 2 * (1 / n_[1] - 1 / n_[0]) / 4
        u = 1 + a
        v_ = theta + b * (1 - tau)
        w_ = theta ** 2 + a * tau * (tau - 2)

        if cont_corr:
            n = n_[1] + n_[0]
            v_ = (v_ - 1 / 2 / n, v_ + 1 / 2 / n)
            w_ = (w_ + 1 / 4 / n ** 2 - theta / n,
                  w_ + 1 / 4 / n ** 2 + theta / n)
        else:
            v_ , w_ = (v_, v_), (w_, w_)

        d_ = (np.sqrt(v_[0] ** 2 - u * w_[0]),
              np.sqrt(v_[1] ** 2 - u * w_[1]))

        low = (v_[0] - d_[0]) / u
        low.clip(lower = -1, inplace = True)

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp = (v_[1] + d_[1]) / u
        upp.clip(upper = 1, inplace = True)

        if self.limits == 'upper':
            return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def haldane(self,
                cont_corr: bool = False) -> pd.DataFrame:
        """Haldane method from Beal's (1987).

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.beal(alpha = -1.0, cont_corr = cont_corr)

    def haldane_cc(self) -> pd.DataFrame:
        """Haldane CI with continuity correction.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.haldane(cont_corr = True)

    def jeffreys_perks(self,
                       cont_corr: bool = False) -> pd.DataFrame:
        """Jeffreys-Perks method from Beal's (1987).

        Returns:
            pd.DataFrame: CI boundaries
        """

        return self.beal(alpha = -0.5, cont_corr = cont_corr)

    def jeffreys_perks_cc(self) -> pd.DataFrame:
        """Jeffreys-Perks CI with continuity correction.

        Returns:
            pd.DataFrame: CI boundaries
        """

        return self.jeffreys_perks(cont_corr = True)


    def brown_li(self) -> pd.DataFrame:
        """'Recentered' CI of Brown & Li (2005). \
            DOI:10.1016/j.jspi.2003.09.039

        Returns:
            pd.DataFrame: CI boundaries.
        """

        n_ = (self.n_0, self.n_1)
        p_ = (self.x_0 / n_[0], self.x_1 / n_[1])

        if self.limits in {'lower', 'upper'}:
            t_conf: float = t.ppf(self.conf, n_[1] + n_[0] - 2)
        else:
            t_conf = t.ppf((1 + self.conf) / 2, n_[1] + n_[0] - 2)

        w = 1 + t_conf ** 2 / (n_[0] + n_[1])
        p: pd.Series = (p_[0] / n_[0] + p_[1] / n_[1]) / \
            (1 / n_[0] + 1 / n_[1])
        delta = p_[1] - p_[0]

        p.clip(
            lower = delta * n_[0] / (n_[0] + n_[1]),
            upper = 1 - delta * n_[1] / (n_[0] + n_[1]),
            inplace = True
        )

        err = np.sqrt(
            w * p * (1 - p) * (1 / n_[0] + 1 / n_[1]) - \
                delta ** 2 / (n_[0] + n_[1])
        )

        low = (delta - t_conf * err) / w

        if self.limits == 'lower':
            return low.to_frame('LL')

        upp = (delta + t_conf * err) / w

        if self.limits == 'upper':
            return upp.to_frame('UL')

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def newcombe(self,
                     cont_corr: bool = False) -> pd.DataFrame:
        """Newcombe (1998)'s hybrid approach, using individual Wilson scores.\
            DOI:10.1002/(sici)1097-0258(19980430)17:8<873::aid-sim779>3.0.co;2-i

        Returns:
            pd.DataFrame: CI boundaries.
        """

        if cont_corr:
            mesial = 'wilson_cc'
        else:
            mesial = 'wilson'

        sample_1 = BinomData(x = self.x_1, n = self.n_1)
        sample_0 = BinomData(x = self.x_0, n = self.n_0)

        p_1 = self.x_1 / self.n_1
        p_0 = self.x_0 / self.n_0

        if self.limits == 'lower':

            rate_1 = Rate(sample_1, 'lower', self.conf)
            rate_0 = Rate(sample_0, 'upper', self.conf)
            w_1 = rate_1.calc_ci(method = mesial)['LL']
            w_0 = rate_0.calc_ci(method = mesial)['UL']

            low = p_1 - p_0 - np.sqrt(
                (p_1 - w_1) ** 2 + (w_0 - p_0) ** 2
            )

            return low.to_frame('LL')

        if self.limits == 'upper':

            rate_1 = Rate(sample_1, 'upper', self.conf)
            rate_0 = Rate(sample_0, 'lower', self.conf)
            w_1 = rate_1.calc_ci(method = mesial)['UL']
            w_0 = rate_0.calc_ci(method = mesial)['LL']

            upp = p_1 - p_0 + np.sqrt(
                (w_1 - p_1) ** 2 + (p_0 - w_0) ** 2
            )

            return upp.to_frame('UL')

        rate_1 = Rate(sample_1, 'both', self.conf)
        rate_0 = Rate(sample_0, 'both', self.conf)
        w_1 = rate_1.calc_ci(method = mesial)
        w_0 = rate_0.calc_ci(method = mesial)

        low = p_1 - p_0 - np.sqrt(
            (p_1 - w_1['LL']) ** 2 + (w_0['UL'] - p_0) ** 2
        )

        upp = p_1 - p_0 + np.sqrt(
            (w_1['UL'] - p_1) ** 2 + (p_0 - w_0['LL']) ** 2
        )

        upp.index = low.index

        return pd.concat([low.rename('LL'),
                          upp.rename('UL')],
                         axis = 1)

    def newcombe_cc(self) -> pd.DataFrame:
        """Newcombe CI from continuity corrected Wilson boundaries.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        return self.newcombe(cont_corr = True)


    def calc_ci(self, method: str, **methods: Any) -> pd.DataFrame:
        """Calculate CI by a particular method.

        Args:
            method (str): One of available methods.

        Returns:
            pd.DataFrame: CI boundaries.
        """

        if method not in RateDiff.__dict__:
            raise ValueError(
                f"Choose 'method' from {list(RateDiff.__dict__.keys())}"
            )

        return RateDiff.__dict__[method](self, **methods)


    def summary(self,
                **methods: dict[str, Any]) -> pd.DataFrame:
        """CI summary by various methods, with horizontal layout."""

        ci_list = []

        if not methods:
            methods = dict.fromkeys(RateDiff.METHODS, {})

        for method, params in methods.items():
            ci_meth = self.calc_ci(method, **params)  # pylint: disable=not-a-mapping
            ci_meth.columns = pd.MultiIndex.from_product(
                [[method], ci_meth.columns]
            )
            ci_list.append(ci_meth)

        ci_smry = pd.concat(ci_list, axis = 1)

        return ci_smry
