import yaml
import pandas as pd
from fastapi import APIRouter #, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from dateutil.relativedelta import relativedelta, MO
from impact2_engine.Enrolment import Enrolment


with open('../impact2_engine/config/config.yaml', 'r') as stream:
    config = yaml.safe_load(stream)


config['data_path'] = '../impact2_engine/data/' + config['data_path']

enr = Enrolment(**config)
enr.preprocess_data()

router = APIRouter(
    prefix="/enrolment",
    tags=["enrolment"],
)


@router.get("/update_data")
async def update_data():
    enr.read_data()
    enr.preprocess_data()

    return {'success': True}



class Metrics(BaseModel):
    site: str
    donor_level: bool
    list_of_metrics: list
    strata: list=[]

@router.post("/get_metrics")
async def get_metrics(metrics: Metrics):
    metrics = metrics.dict()
    metrics_df = enr.get_enrolment_rate_metrics(**metrics)

    activation_date = enr.get_activation_date(metrics['site'])

    metrics_df = metrics_df.merge(activation_date)

    return metrics_df.to_dict(orient = 'records')



class Projections(BaseModel):
    max_weeks: int
    target_sample: int
    columns_to_use: list
    donor_level: bool
    projection_steps: dict

@router.post("/get_projections")
async def get_projections(projections: Projections):
    projections = projections.dict()
    proj = enr.get_projections(**projections)

    columns_to_use = projections['columns_to_use']

    if len(columns_to_use) == 2:
        temp_df_0 = proj.groupby(['week', columns_to_use[0]])['value'].sum().reset_index()
        temp_df_0['week'] = temp_df_0['week'].apply(lambda x: x.normalize()).astype(str)
        temp_df_1 = proj.groupby(['week', columns_to_use[1]])['value'].sum().reset_index()
        temp_df_1['week'] = temp_df_1['week'].apply(lambda x: x.normalize()).astype(str)

        temp_proj = {}
        temp_proj[columns_to_use[0]] = temp_df_0.to_dict(orient = 'records')
        temp_proj[columns_to_use[1]] = temp_df_1.to_dict(orient = 'records')
    else:
        temp_proj = proj.to_dict(orient = 'records')

    return temp_proj


class ModelProjections(BaseModel):
    max_weeks: int
    target_sample: int
    columns_to_use: list
    donor_level: bool


@router.post("/get_model_projections")
async def get_model_projections(model_projections: ModelProjections):
    model_projections = model_projections.dict()
    model_projections = enr.get_model_projections(**model_projections)

    return model_projections.to_dict(orient = 'records')


class BarChart(BaseModel):
    show_by: str
    by_donor: bool
    stratify_by: list
    only_new_donors: bool=False


@router.post("/get_bar_chart")
async def get_bar_chart(bar_chart: BarChart):
    bar_chart = bar_chart.dict()
    bar_chart_df = enr.get_bar_chart(**bar_chart)

    show_by = bar_chart['show_by']
    stratify_by = bar_chart['stratify_by']

    if show_by == 'day':
        freq = 'D'
        date_range_df = pd.Series(pd.date_range(pd.to_datetime(bar_chart_df.reset_index()[show_by]).min(), pd.to_datetime(bar_chart_df.reset_index()[show_by]).max(), freq=freq, inclusive='both'))
        date_range_df.name = show_by
    elif show_by == 'week':
        freq = 'W'
        date_range_df = pd.Series(pd.date_range(pd.to_datetime(bar_chart_df.reset_index()[show_by]).min(), pd.to_datetime(bar_chart_df.reset_index()[show_by]).max()+ relativedelta(weekday=MO(+2)), freq=freq, inclusive='both'))
        date_range_df.name = show_by
        date_range_df = date_range_df.apply(lambda x: x + relativedelta(weekday=MO(-1)))

    dfs = []
    for i,j in bar_chart_df.groupby(stratify_by):
        merged = j.merge(date_range_df.apply(lambda x: x.normalize()).astype(str), on = show_by, how='outer')
        merged[['absolute', 'percentage']] = merged[['absolute', 'percentage']].fillna(0)
        merged[stratify_by] = merged[stratify_by].ffill()
        merged = merged.sort_values(by = show_by, ascending=True)

        merged[['cumsum']] = merged[['cumsum']].ffill()
        merged[['cumsum']] = merged[['cumsum']].fillna(0)
        dfs.append(merged)

    temp = pd.concat(dfs).reset_index(drop=True)
    temp = temp.pivot_table(index = [show_by], columns = stratify_by, values = ['cumsum', 'absolute', 'percentage'])

    temp = temp.melt(ignore_index=False).reset_index().rename(columns = {None: 'metric'})

    temp = temp.to_dict(orient='records')

    return temp
