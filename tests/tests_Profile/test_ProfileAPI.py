"""Unit tests for SafetyAPI."""

import unittest
from typing import Any
from fastapi import FastAPI
from fastapi.testclient import TestClient
from impact2_engine.Profile import ProfileAPI


app = FastAPI()
app.include_router(ProfileAPI.router)
client = TestClient(app)


class TestProfileAPI(unittest.TestCase):
    """Unit tests for SafetyAPI."""

    def test_profile_update_data(self) -> None:
        """Update data completion."""

        result = client.get('/profile/update_data')

        self.assertTrue(result.json()['success'])


    def test_get_demographics(self) -> None:
        """Get demographics validation."""

        param = {'donor_id': ['383902', '439443']}

        result = client.post(
            '/profile/get_demographics', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_collection_summary(self) -> None:
        """Get collection summary validation."""

        param = {'donor_id': ['383902', '439443']}

        result = client.post(
            '/profile/get_collection_summary', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')


    def test_get_chronology(self) -> None:
        """Get chronology validation."""

        param = {'donor_id': ['383902', '439443']}

        result = client.post(
            '/profile/get_chronology', json = param
        )

        self.assertTrue(
            result.status_code == 200, msg = 'No response'
        )

        result_json: list[dict[str, Any]] = result.json()

        self.assertTrue(result_json, msg = 'Empty result')
