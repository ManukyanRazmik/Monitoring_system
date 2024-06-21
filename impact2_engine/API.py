"""Main API"""

from fastapi import FastAPI #, Depends, Header, HTTPException
import uvicorn

from Enrolment import EnrolmentAPI
from PlasmaVolume import PlasmaVolumeAPI
from impact2_engine.Safety import SafetyAPI
from impact2_engine.PlasmaCollection import PlasmaCollectionAPI
from impact2_engine.Profile import ProfileAPI
from impact2_engine.Power import PowerAPI
from impact2_engine.Dashboard import DashboardAPI


app = FastAPI()
app.include_router(EnrolmentAPI.router)
app.include_router(PlasmaVolumeAPI.router)
app.include_router(SafetyAPI.router)
app.include_router(PlasmaCollectionAPI.router)
app.include_router(ProfileAPI.router)
app.include_router(PowerAPI.router)
app.include_router(DashboardAPI.router)

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)