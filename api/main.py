"""FastAPI service: `uvicorn api.main:app` -> interactive docs at /docs"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from rentcast.predict import RentModel  # noqa: E402

app = FastAPI(title="rentcast", version="0.1.0")
model = RentModel.load(ROOT / "models" / "rent_model.joblib")


class Listing(BaseModel):
    state: str = Field(examples=["TX"])
    city: str | None = Field(default=None, examples=["Austin"])
    lat: float | None = None
    lon: float | None = None
    square_feet: float = Field(ge=200, le=5000)
    bedrooms: float = Field(ge=0, le=8)
    bathrooms: float = Field(ge=1, le=8)
    amenities: list[str] = []
    pets: str = "None"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return model.metrics


@app.post("/predict")
def predict(listing: Listing):
    try:
        return model.predict(**listing.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
