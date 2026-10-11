"""
Unit and integration tests for Battery Management System API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_soh_predict(client):
    payload = {
        "battery_id": "B0005",
        "voltage": [4.19, 4.15, 4.10, 4.05, 4.00, 3.95, 3.90],
        "current": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        "temperature": [24.5, 24.8, 25.1, 25.3, 25.5, 25.8, 26.0],
        "time": [0, 60, 120, 180, 240, 300, 360],
    }
    response = client.post("/api/v1/soh/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "soh_percent" in data
    assert 50.0 <= data["soh_percent"] <= 100.0
    assert "ecm_params" in data
    assert "R0" in data["ecm_params"]


def test_soc_predict(client):
    payload = {
        "battery_id": "B0005",
        "voltage": [4.15, 4.10, 4.05, 4.00],
        "current": [1.5, 1.5, 1.5, 1.5],
        "temperature": [25.0, 25.1, 25.2, 25.3],
        "time": [0, 100, 200, 300],
        "initial_soc": 95.0,
    }
    response = client.post("/api/v1/soc/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "soc_trajectory" in data
    assert len(data["soc_trajectory"]) == 4
    assert "current_soc" in data


def test_rul_predict(client):
    payload = {
        "battery_id": "B0005",
        "voltage": [
            4.18, 4.12, 4.08, 4.04, 4.00, 3.96, 3.92, 3.88, 3.84, 3.80,
            3.76, 3.72, 3.68, 3.64, 3.60, 3.56, 3.52, 3.48, 3.44, 3.40,
            3.36, 3.32, 3.28, 3.24, 3.20, 3.15, 3.10, 3.00, 2.90, 2.80
        ],
        "current": [2.0] * 30,
        "temperature": [24.5 + i * 0.45 for i in range(30)],
        "current_cycle": 50,
        "capacity": 1.75,
        "nominal_capacity": 2.0,
    }
    response = client.post("/api/v1/rul/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "rul_cycles" in data
    assert data["rul_cycles"] > 0
    assert "confidence_interval" in data
    assert "prediction_source" in data


def test_ecm_analyze(client):
    payload = {
        "battery_id": "B0005",
        "model_type": "RC",
        "voltage": [4.2, 4.15, 4.1, 4.05, 4.0],
        "current": [1.0, 1.0, 1.0, 1.0, 1.0],
        "time": [0, 10, 20, 30, 40],
    }
    response = client.post("/api/v1/ecm/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "parameters" in data
    assert "metrics" in data
    assert "voltage_predicted" in data


def test_fault_detect(client):
    payload = {
        "battery_id": "B0005",
        "current_params": {
            "R0": 0.080,  # +77% over healthy baseline 0.045 -> critical fault
            "R1": 0.022,
            "C1": 1400.0,
        },
    }
    response = client.post("/api/v1/fault/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faults_detected" in data
    assert len(data["faults_detected"]) > 0
    assert data["overall_health"] in ["warning", "critical"]


def test_audit_history(client):
    # Verify that prediction audit log records are queryable
    response = client.get("/api/v1/audit/history?battery_id=B0005")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["battery_id"] == "B0005"
