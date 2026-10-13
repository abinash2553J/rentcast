import pathlib

import pytest

pytestmark = pytest.mark.skipif(not pathlib.Path("models/rent_model.joblib").exists(), reason="model not trained")


def test_api_roundtrip():
    from fastapi.testclient import TestClient

    from api.main import app, model

    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    state = next(iter(model.cities))
    city = next(iter(model.cities[state]))
    r = c.post("/predict", json={"state": state, "city": city, "square_feet": 900, "bedrooms": 2, "bathrooms": 1})
    assert r.status_code == 200 and r.json()["low"] <= r.json()["mid"] <= r.json()["high"]
    assert c.post("/predict", json={"state": state, "city": "Nowhere", "square_feet": 900, "bedrooms": 2, "bathrooms": 1}).status_code == 422
    assert c.post("/predict", json={"state": state, "city": city, "square_feet": 10, "bedrooms": 2, "bathrooms": 1}).status_code == 422
