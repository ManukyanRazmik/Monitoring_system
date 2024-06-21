"""Unit tests for SafetyAPI."""

import unittest
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from impact2_engine.PlasmaCollection import PlasmaCollectionAPI

app = FastAPI()
app.include_router(PlasmaCollectionAPI.router)
client = TestClient(app)


class TestPlasmaCollectionAPI(unittest.TestCase):
    """Unit tests for SafetyAPI."""

    def test_plasma_update_data(self) -> None:
        """Update data completion."""

        result = client.get('/plasma/update_data')

        self.assertTrue(result.json()['success'])


    def test_get_summary(self) -> None:
        """Validate plasma collection summary."""

        # result_get = client.get('/plasma/get_summary')

        # self.assertTrue(
        #     result_get.status_code == 200, msg = 'No response'
        # )

        # result_json: list[dict[str, Any]] = result_get.json()

        # self.assertTrue(result_json, msg = 'Empty result')

        param = {
            'population': 'itt',  # default
            'strata': ['site']    # default
        }

        result_post = client.post('/plasma/get_summary', json = param)

        self.assertTrue(
            result_post.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result_post.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_distribution(self) -> None:
        """Validate plasma metrics distribution."""

        # result = client.get('/plasma/get_distribution')

        # self.assertTrue(
        #     result.status_code == 200, msg = 'No response'
        # )

        # result_json: list[dict[str, Any]] = result.json()

        # self.assertTrue(result_json, msg = 'Empty result')

        param = {
            'population': 'itt',  # default
            'strata': ['site']      # default
        }

        result_post = client.post('/plasma/get_distribution', json = param)

        self.assertTrue(
            result_post.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result_post.json()

        self.assertTrue(result_json, msg = 'Empty result')