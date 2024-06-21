import yaml
import pandas as pd
from fastapi import APIRouter #, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
# from dateutil.relativedelta import relativedelta, MO
from impact2_engine.Dashboard import Dashboard


with open('../impact2_engine/config/dashboard.yml', 'r') as stream:
	config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

dash = Dashboard(**config)

router = APIRouter(
	prefix="/dashboard",
	tags=["dashboard"],
)


@router.get("/update_data")
async def update_data():
	dash.update_data()

	return {'success': True}



@router.get("/donations")
async def donations():
	demo_table = dash.summary_donation()

	return demo_table.to_dict()


class Demography(BaseModel):    
	donor_level: bool
	

@router.post("/demographics")
async def demographics(demography: Demography):
	tables = {}
	d_level = demography.dict()
	psble_groups = [col['name'] for col in config['contents']['CAT']]
	for grouping in psble_groups:
		demog = dash.summary_demographic(by = grouping, **d_level)
		demog = demog.to_dict()
		tables[grouping.lower()] = demog

	return tables



@router.get("/metrics")
async def metrics():
	metric_table = dash.summary_metrc()

	return metric_table.to_dict()



