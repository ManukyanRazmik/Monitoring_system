"""Safety API"""

from typing import Optional, Union, Any
import datetime as dt
import pandas as pd
import numpy as np
import yaml
from fastapi import APIRouter
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from impact2_engine.Safety.Safety import Safety


with open('../impact2_engine/config/safety_config.yml',
          mode = 'r', encoding = 'utf-8') as stream:
    config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

saf = Safety(**config)

router = APIRouter(
    prefix = "/safety",
    tags = ["safety"],
)


@router.get("/update_data/")
async def update_safety() -> dict[str, bool]:
    """Check, load and preprocess data at request"""

    saf.update_data()

    return {'success': True}


@router.get("/get_raw_data/")
async def get_raw_data() -> Any:
    """Get raw data, only relevant columns."""

    raw_data = saf.data.reset_index()

    return raw_data.to_dict(orient = 'records')


@router.get("/get_general_summary/")
async def get_general_summary() -> Any:
    """General summary in the longer format, split by 'group'."""

    smry_don: pd.DataFrame = saf.summary_donat()
    smry_rsk: pd.DataFrame = saf.summary_risk()
    smry_rsk.loc[:, ('ser_hyp', 'sum')] = np.nan
    smry_rsk.loc[:, ('ser_hyp', 'pct')] = np.nan

    ## Flattened nested format
    smry_rsk = smry_rsk.stack('variable', dropna = False)\
        .replace({np.nan: None})
    smry_rsk.columns.name = None
    smry_rsk.reset_index(inplace = True)

    ## Glue names, keeping wide format
    # smry_gen = smry_gen.stb.flatten(levels = [1, 0], reset = True)
    # smry_rsk = smry_rsk.stb.flatten(levels = [1, 0], reset = True)

    ## Convert to longer representation
    smry_don = smry_don.stack([0, 1]).replace({np.nan: None})
    smry_don = smry_don.reset_index(name = 'value')
    # smry_rsk = smry_rsk.stack([0, 1]).dropna()
    # smry_rsk = smry_rsk.reset_index(name = 'value')

    # Either join, or return as two-table dictionary

    out_dict: dict[str, Any] = {
        'donation': smry_don.to_dict(orient = 'records'),
        'risk': smry_rsk.to_dict(orient = 'records')
    }

    return out_dict


class SevPop(BaseModel):
    """Filter param: severity/population."""
    severity: str
    population: str = 'itt'

@router.post("/get_event_summary/")
async def get_event_summary(custom: SevPop) -> Any:
    """Individual AE summary, for chosen POP and SEV classification."""

    param = {'sev': custom.severity,
             'pop': custom.population}

    smry_tbl = saf.summary_aes(**param, conditional = False).stack('variable')
    smry_tbl.rename_axis(None, axis = 1, inplace = True)
    smry_tbl.reset_index(inplace = True)
    smry_tbl.replace(
        to_replace = {'pct': {np.nan: None}}, inplace = True
    )

    return smry_tbl.to_dict(orient = 'records')

@router.post("/get_event_piechart/")
async def get_event_piechart(custom: SevPop) -> Any:
    """Individual AE summary, for chosen POP and SEV classification."""

    param = {'sev': custom.severity,
             'pop': custom.population}

    smry_tbl = saf.summary_aes(**param, conditional = True)\
        .stack('variable').loc['grand_total', ]
    smry_tbl.rename_axis(None, axis = 1, inplace = True)
    smry_tbl.reset_index(inplace = True)
    smry_tbl.replace(
        to_replace = {'pct': {np.nan: None}}, inplace = True
    )

    return smry_tbl.to_dict(orient = 'records')


class SevPopStrata(BaseModel):
    """Filter severity/population, split by strata."""
    strata: list[str]
    severity: str
    population: str = 'itt'

@router.post("/get_severity_piechart/")
async def get_severity_piechart(custom: SevPopStrata) -> Any:
    """SEV counts, split by 'group' + single 'strata'."""

    if len(custom.strata) != 1:
        raise KeyError(
            """'strata' should be list[str] of length 1,
            containing SINGLE categorical type in addition to 'group'.
            """
        )

    param: dict[str, Any] = {
        'strata': ['group'] + custom.strata,
        'sev': custom.severity,
        'pop': custom.population
    }

    smry_tbl = saf.summary_sev(**param)

    smry_tbl.columns = smry_tbl.columns.get_level_values("metric")

    smry_tbl.loc[[('A', 'A - subtotal'), ('B', 'B - subtotal')], 'pct'] = \
        saf.data.groupby('group')[param['sev']].sum()\
            .transform(lambda x: x/x.sum() * 100).values
    smry_tbl.loc[('grand_total', ' '), 'pct'] = None

    smry_tbl.rename_axis(None, axis = 1, inplace = True)
    smry_tbl.reset_index(inplace = True)
    smry_tbl.replace(
        to_replace = {'pct': {np.nan: None}}, inplace = True
    )

    return smry_tbl.to_dict(orient = 'records')


@router.post("/get_severity_bargraph/")
async def get_severity_bargraph(custom: SevPopStrata) -> Any:
    """SEV counts, split by 'strata', constrained to date range.
    % normalization for the innermost (last) stratification."""

    param: dict[str, Any] = {
        'strata': custom.strata,
        'sev': custom.severity,
        'pop': custom.population
    }

    smry_tbl = saf.summary_sev(**param)

    smry_tbl.columns = smry_tbl.columns.get_level_values("metric")

    return smry_tbl.reset_index().to_dict(orient = 'records')


class CI(BaseModel):
    """Confidence interval parameters + filter."""
    severity: str
    population: str = 'itt'
    limits: str = 'both'
    signif_level: float = .05

@router.post("/get_confidence_interval/")
async def get_confidence_interval(custom: CI) -> Any:
    """CI boundaries by different methods."""

    param: dict[str, Any] = {
        'sev': custom.severity,
        'pop': custom.population,
        'limits': custom.limits,
        'conf': 1 - custom.signif_level
    }

    ci_smry = saf.calc_ci(**param)

    ci_smry.reset_index(names = 'method', inplace = True)

    return ci_smry.to_dict(orient = 'records')


class PeriodCI(BaseModel):
    """Longitudinal summary parameters."""
    method: str
    severity: str
    population: str = 'itt'
    aggregate: str = 'w'
    signif_level: float = .05
    start: Optional[Union[dt.date, str]] = None
    end: Optional[Union[dt.date, str]] = None


@router.post("/get_longitudinal_summary/")
async def get_longitudinal_summary(custom: PeriodCI) -> Any:
    """CI boundaries by different methods."""

    param: dict[str, Any] = {
        'method': custom.method,
        'sev': custom.severity,
        'pop': custom.population,
        'aggregate': custom.aggregate,
        'conf': 1 - custom.signif_level,
        'start': custom.start,
        'end': custom.end
    }

    ci_smry = saf.summary_longitudinal(**param)

    ci_smry = ci_smry.stack([0, 1])
    ci_smry = ci_smry.reset_index(name = 'value')

    return ci_smry.to_dict(orient = 'records')
