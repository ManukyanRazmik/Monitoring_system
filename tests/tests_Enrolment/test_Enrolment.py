import unittest
from impact2_engine.Enrolment import EnrolmentAPI
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()
app.include_router(EnrolmentAPI.router)
client = TestClient(app)


class TestEnrolmentAPI(unittest.TestCase):

    def test_enrolmentapi_update_data(self):
        result = client.get('/enrolment/update_data')

        self.assertTrue(result.json()['success'])


    # def test


