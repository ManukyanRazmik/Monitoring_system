"""Safety API"""

from typing import Any
import pandas as pd
import yaml
from fastapi import APIRouter
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from impact2_engine.PlasmaCollection.PlasmaCollection import PlasmaCollection


with open('../impact2_engine/config/plasma_config.yml',
          mode = 'r', encoding = 'utf-8') as stream:
    config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

col = PlasmaCollection(**config)

router = APIRouter(
    prefix = "/plasma_collection",
    tags = ["plasma_collection"],
)


@router.get("/update_data/")
async def update_plasma() -> dict[str, bool]:
    """Check, load and preprocess data at request"""

    col.update_data()

    return {'success': True}


class PopStrata(BaseModel):
    """Filter population."""
    population: str = 'itt'
    strata: list[str] = ['site']


@router.post("/get_summary/")
async def get_summary(custom: PopStrata) -> Any:
    """General summary information. The whole range by default."""

    param: dict[str, Any] = {
        'pop': custom.population,
        'strata': custom.strata
    }

    overall = param.copy()
    overall['strata'] = None

    count_split = col.count_summary(**param)
    count_all = col.count_summary(**overall)
    count_all[param['strata']] = 'overall'

    plasma_split = col.plasma_summary(**param)
    plasma_all = col.plasma_summary(**overall)
    plasma_all[param['strata']] = 'overall'

    smry = pd.concat(
        [count_split, count_all,
         plasma_split, plasma_all],
        ignore_index = True
    )

    return smry.to_dict(orient = 'records')


@router.post("/get_distribution/")
async def get_distribution(custom: PopStrata) -> Any:
    """General summary information. The whole range by default."""

    param: dict[str, Any] = {
        'pop': custom.population,
        'strata': custom.strata
    }

    overall = param.copy()
    overall['strata'] = None

    plasma_split = col.plasma_summary(**param, raw = True)
    plasma_all = col.plasma_summary(**overall, raw = True)
    plasma_all[param['strata']] = 'overall'

    smry = pd.concat(
        [plasma_split, plasma_all],
        ignore_index = True
    )

    return smry.to_dict(orient = 'records')
