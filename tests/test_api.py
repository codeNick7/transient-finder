"""Unit tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert "version" in data


def test_api_fixtures():
    response = client.get("/api/candidates/fixtures")
    assert response.status_code == 200
    fixtures = response.json()
    assert len(fixtures) >= 5
    ids = [f["candidate_id"] for f in fixtures]
    assert "NEO_APOLLO_REAL_01" in ids


def test_api_validate_endpoint():
    # Fetch a fixture and post to /api/validate
    fixtures = client.get("/api/candidates/fixtures").json()
    sample = fixtures[0]
    
    response = client.post("/api/validate", json=sample)
    assert response.status_code == 200
    result = response.json()
    assert result["candidate_id"] == sample["candidate_id"]
    assert "status" in result
    assert "composite_confidence_score" in result
    assert "heuristic_breakdown" in result


def test_api_revalidate_endpoint():
    fixtures = client.get("/api/candidates/fixtures").json()
    sample = next(f for f in fixtures if "2026" in f["candidate_id"])
    
    response = client.post("/api/revalidate", json=sample)
    assert response.status_code == 200
    report = response.json()
    assert report["candidate_id"] == sample["candidate_id"]
    assert "final_disposition" in report
    assert "recovery_probability" in report


def test_api_feature_importance():
    response = client.get("/api/model/importance")
    assert response.status_code == 200
    data = response.json()
    assert "feature_importances" in data
    assert len(data["feature_importances"]) > 10
