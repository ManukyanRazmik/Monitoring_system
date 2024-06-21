"""Safety API"""

from typing import Union, Any
import datetime as dt
import pandas as pd
import numpy as np
import yaml
from fastapi import APIRouter
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from impact2_engine.Power.Power import Power


with open('../impact2_engine/config/power_config.yml',
          mode = 'r', encoding = 'utf-8') as stream:
    config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

pwr = Power(**config)

router = APIRouter(
    prefix = "/power",
    tags = ["power"],
)


@router.get("/update_data/")
async def update_power() -> dict[str, bool]:
    """Check and load data at request"""

    pwr.update_data()

    return {'success': True}



class Prior(BaseModel):
    """Set prior user expetations."""
    prior_p: list[float] = [.5, .5]
    prior_n: list[int] = [1, 1]
    severity: str = 'sig_hyp'
    time_point: Union[dt.date, str] | None = None


@router.post("/get_posterior/")
async def get_posterior(custom: Prior) -> Any:
    """Beta posterior distribution for binomial conjugate model."""

    param = custom.dict()

    smry_tbl = pwr.posterior(**param)
    smry_tbl.replace({np.nan: None}, inplace = True)

    return smry_tbl.to_dict(orient = 'records')


class Setup(Prior):
    """Set prior user expetations."""
    samples: int = 50000
    step: int = 1000
    alpha: float = .05
    method: str = 'wald_cc'
    prior_p: list[float] = [.5, .5]
    prior_n: list[int] = [1, 1]
    severity: str = 'sig_hyp'
    time_point: Union[dt.date, str] | None = None


@router.post("/get_simulation/")
async def get_simulation(custom: Setup) -> Any:
    """Use posterior to simulate the AE for range of sample sizes."""

    param = custom.dict()

    smry_tbl = pwr.simulate(**param)
    smry_tbl.reset_index(inplace = True, names = 'group_size')
    smry_tbl.replace({np.nan: None}, inplace = True)

    return smry_tbl.to_dict(orient = 'records')
