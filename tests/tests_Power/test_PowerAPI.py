"""Unit tests for SafetyAPI."""

import unittest
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from impact2_engine.Power import PowerAPI


app = FastAPI()
app.include_router(PowerAPI.router)
client = TestClient(app)


class TestPowerAPI(unittest.TestCase):
    """Unit tests for SafetyAPI."""

    def test_power_update_data(self) -> None:
        """Update data completion."""

        result = client.get('/power/update_data')

        self.assertTrue(result.json()['success'])


    def test_get_posterior(self) -> None:
        """Get demographics validation."""

        param = {
            'prior_p': [.5, .5],
            'prior_n': [1, 1],
            'severity': 'sig_hyp',
            'time_point': None
        }

        result = client.post(
            '/power/get_posterior', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_simulation(self) -> None:
        """Get collection summary validation."""

        param = {
            'samples': 10000,
            'step': 1000,
            'alpha': .05,
            'method': 'wald_cc',
            'prior_p': [.5, .5],
            'prior_n': [1, 1],
            'severity': 'sig_hyp',
            'time_point': None
        }

        result = client.post(
            '/power/get_simulation', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')
