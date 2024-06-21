"""Safety API"""

from typing import Any
import pandas as pd
import numpy as np
import yaml
from fastapi import APIRouter
from pydantic import BaseModel  # pylint: disable=no-name-in-module
from impact2_engine.Profile.Profile import Profile


with open('../impact2_engine/config/profile_config.yml',
          mode = 'r', encoding = 'utf-8') as stream:
    config = yaml.safe_load(stream)

config['data_path'] = '../impact2_engine/data/' + config['data_path']

full_names = {
    # collections table
    'diff_bmi': 'first_bmi',
    'diff_hct': 'first_hct',
    'diff_weight': 'diff_weight',
    'mean_duration_minutes': 'mean_duration_minutes',
    'nunique_col_id': 'n_collections',
    'mean_speed': 'mean_speed',
    'sum_all_ae': 'n_all_ae',
    'sum_sig_hyp': 'n_sig_hyp',
    # chronology table
    'don_id': 'donor_id',
    'col_id': 'collection_id',
    'dev_id': 'device_id',
    'col_date': 'collection_date',
    'proc_start': 'procedure_start',
    'proc_end': 'procedure_end',
    'duration_minutes': 'duration_minutes',
    'speed': 'speed',
    'yield': 'yield',
    'target_vol': 'target_volume',
    'actual_vol': 'actual_volume',
    'weight': 'weight',
    'bmi': 'bmi',
    'hct': 'hct',
    'days_total': 'days_total',
    'days_lag': 'days_lag',
    'AE': 'event_type'
}

prof = Profile(**config)

router = APIRouter(
    prefix = "/profile",
    tags = ["profile"],
)


@router.get("/update_data/")
async def update_profile() -> dict[str, bool]:
    """Check and load data at request"""

    prof.update_data()

    return {'success': True}



class DonorID(BaseModel):
    """List of donor IDs, include all by default."""
    donor_id: list[str] | None = None


@router.post("/get_demographics/")
async def get_demographics(custom: DonorID) -> Any:
    """Individual AE summary, for chosen POP and SEV classification."""

    param = {'don_ids': custom.donor_id}

    smry_tbl = prof.summary_dem(**param).reset_index()
    smry_tbl.replace({np.nan: None}, inplace = True)

    return smry_tbl.to_dict(orient = 'records')


@router.post("/get_collection_summary/")
async def get_collection_summary(custom: DonorID) -> Any:
    """Individual AE summary, for chosen POP and SEV classification."""

    param = {'don_ids': custom.donor_id}

    smry_tbl = prof.summary_col(**param).droplevel(0, axis = 1)
    smry_tbl.reset_index(inplace = True)
    smry_tbl.replace({np.nan: None}, inplace = True)

    return smry_tbl.to_dict(orient = 'records')


@router.post("/get_chronology/")
async def get_chronology(custom: DonorID) -> Any:
    """Individual AE summary, for chosen POP and SEV classification."""

    param = {'don_ids': custom.donor_id}

    smry_tbl = prof.chronology(**param)
    smry_tbl.replace({np.nan: None}, inplace = True)

    smry_tbl.columns = [full_names[col] for col in smry_tbl.columns]

    return smry_tbl.to_dict(orient = 'records')
