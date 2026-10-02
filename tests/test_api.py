"""Unit tests for FastAPI REST Endpoints."""

from fastapi.testclient import TestClient

from finsight.api.server import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "components" in data


def test_companies_endpoint():
    response = client.get("/companies")
    assert response.status_code == 200
    data = response.json()
    assert "primary_target" in data
    assert data["primary_target"]["ticker"] == "TATAMOTORS.NS"


def test_evaluation_summary_endpoint():
    response = client.get("/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "classical_ml" in data
    assert "deep_learning" in data
    assert "nlp_sentiment" in data
    assert "filings_rag" in data


def test_post_analyze_and_get_job():
    post_res = client.post(
        "/analyze",
        json={"query": "Analyze Tata Motors for the next 12 months", "horizon_months": 12},
    )
    assert post_res.status_code == 200
    job_info = post_res.json()
    assert "job_id" in job_info
    job_id = job_info["job_id"]

    # Fetch result
    get_res = client.get(f"/jobs/{job_id}")
    assert get_res.status_code == 200
    result = get_res.json()
    assert result["job_id"] == job_id
    assert "final_memo_markdown" in result
    assert "charts" in result
