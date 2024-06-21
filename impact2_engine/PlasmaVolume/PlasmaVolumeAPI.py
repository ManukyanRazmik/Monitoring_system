from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
import uvicorn
import yaml
import os

from impact2_engine.PlasmaVolume import PlasmaVolume
import datetime as dt
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta, MO



with open('../impact2_engine/config/PlasmaVolume.yaml', 'r') as stream:
    config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

pv = PlasmaVolume(**config)


router = APIRouter(
    prefix="/plasma_volume",
    tags=["plasma_colume"],
)


@router.get("/update_data")
async def update_data():
    pv.update_data()

    return {'success': True}


@router.get("/get_plasma_collection_summary")
async def get_plasma_collection_summary():
    summary = pv.get_plasma_collection_summary()

    return summary.to_dict(orient = 'records')



class ComparativeGain(BaseModel):
    population: str
    stratify_by: str

@router.post("/get_comparative_gain")
async def get_comparative_gain(comparative_gain: ComparativeGain):

    comparative_gain_df = pv.get_comparative_gain(**comparative_gain.dict())

    return comparative_gain_df.to_dict(orient = 'records')



class LongitudinalStat(BaseModel):
    population: str
    show_by: str

@router.post("/get_longitudinal_stat")
async def get_longitudinal_stat(longitudinal_stat: LongitudinalStat):

    longitudinal_stat_df = pv.get_longitudinal_stat(**longitudinal_stat.dict())

    return longitudinal_stat_df.to_dict(orient = 'records')