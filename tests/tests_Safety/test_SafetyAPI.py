"""Unit tests for SafetyAPI."""

import unittest
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from impact2_engine.Safety import SafetyAPI
from impact2_engine.utils.CalcCI import Rate

app = FastAPI()
app.include_router(SafetyAPI.router)
client = TestClient(app)


class TestSafetyAPI(unittest.TestCase):
    """Unit tests for SafetyAPI."""

    def test_safety_update_data(self) -> None:
        """Update data completion."""

        result = client.get('/safety/update_data')

        self.assertTrue(result.json()['success'])


    def test_get_raw_data(self) -> None:
        """Get raw data response validation."""

        result = client.get('/safety/get_raw_data')

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_general_summary(self) -> None:
        """Get raw data validation."""

        result = client.get('/safety/get_general_summary')

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: dict[str, list[dict]] = result.json()

        self.assertIn(
            'donation', result_json, msg = 'Missing donation'
        )
        self.assertIn(
            'risk', result_json, msg = 'Missing risk'
        )

        self.assertTrue(
            result_json['donation'], msg = 'Empty donation'
        )

        self.assertTrue(
            result_json['risk'], msg = 'Empty risk'
        )


    def test_get_event_summary(self) -> None:
        """Get event summary validation."""

        param = {'severity': 'non_hyp', 'population': 'pp'}

        result = client.post(
            '/safety/get_event_summary', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_severity_piechart(self) -> None:
        """Get severity piechart validation."""

        param = {
            'severity': 'non_hyp', 'population': 'mitt',
            'strata': ['gender']
        }

        result = client.post(
            '/safety/get_severity_piechart', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_severity_bargraph(self) -> None:
        """Get severity bargraph validation."""

        param = {
            'severity': 'non_hyp', 'population': 'mitt',
            'strata': ['site', 'group', 'gender']
        }

        result = client.post(
            '/safety/get_severity_bargraph', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_confidence_interval(self) -> None:
        """Get confidence interval validation."""

        param = {
            'limits': 'both',
            'severity': 'non_hyp',
            'population': 'mitt',
            'signif_level': .05
        }

        result = client.post(
            '/safety/get_confidence_interval', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response: 2-sided'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result: 2-sided')

        param['limits'] = 'upper'

        self.assertTrue(
            result.status_code == 200, msg = 'No response: 1-sided'
        )

        result_json = result.json()

        self.assertTrue(result_json, msg = 'Empty result: 1-sided')


    def test_get_longitudinal_summary(self) -> None:
        """Get longitudinal summary validation."""

        param = {
                'aggregate': 'w',
                'severity': 'non_hyp',
                'population': 'mitt',
                'signif_level': .05,
                'start': '2020-02-01',
                'end': None
            }

        for method in Rate.METHODS:

            param['method'] = method

            result = client.post(
                '/safety/get_longitudinal_summary', json = param
            )

            self.assertTrue(
                result.status_code == 200,
                msg = f"No response {method}"
            )

            result_json: list[dict[str, Any]] = result.json()

            self.assertTrue(
                result_json, msg = f"Empty result: {method}"
            )
