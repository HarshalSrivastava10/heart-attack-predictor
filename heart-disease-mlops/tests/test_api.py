from __future__ import annotations

from fastapi.testclient import TestClient


def test_health():
    from api.main import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_with_model(tmp_model_bundle, sample_features):
    from api.main import app

    client = TestClient(app)
    r = client.post("/predict", json=sample_features)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probability_positive" in body
    assert "confidence" in body
    assert "risk_level" in body
    assert body["risk_level"] in (
        "no risk",
        "low risk",
        "medium risk",
        "high risk",
    )
    assert body["prediction"] in (0, 1)


def test_metrics_endpoint():
    from api.main import app

    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
