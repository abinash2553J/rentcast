"""Inference wrapper shared by the Streamlit app and the API."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import build_features


class RentModel:
    def __init__(self, bundle: dict):
        self.models, self.columns = bundle["models"], bundle["columns"]
        self.cities, self.metrics = bundle["cities"], bundle["metrics"]
        self.pad = bundle.get("pad", 0.0)
        self.amenities = bundle["amenities"]  # raw dataset names, e.g. "Washer Dryer"

    @classmethod
    def load(cls, path: str | Path = "models/rent_model.joblib") -> "RentModel":
        return cls(joblib.load(path))

    def predict(self, *, state: str, square_feet: float, bedrooms: float, bathrooms: float,
                city: str | None = None, lat: float | None = None, lon: float | None = None,
                amenities: list[str] | tuple[str, ...] = (), pets: str = "None") -> dict:
        """Return {low, mid, high} monthly rent in USD (~80% interval). Location = city, or explicit lat/lon."""
        if lat is None or lon is None:
            try:
                lat, lon = self.cities[state][city]
            except KeyError:
                raise ValueError(f"unknown location: city={city!r}, state={state!r}") from None
        if not 200 <= square_feet <= 5000:
            raise ValueError("square_feet must be between 200 and 5000")
        row = pd.DataFrame([{
            "square_feet": square_feet, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "latitude": lat, "longitude": lon, "state": state,
            "amenities": ",".join(amenities), "pets_allowed": pets,
        }])
        X = build_features(row).reindex(columns=self.columns, fill_value=0)
        log = {k: float(m.predict(X)[0]) for k, m in self.models.items()}
        mid = np.exp(log["mid"])
        # calibrated interval; quantile models can cross, so clamp around the median
        low = min(np.exp(log["low"] - self.pad), mid)
        high = max(np.exp(log["high"] + self.pad), mid)
        return {"low": round(low), "mid": round(mid), "high": round(high), "lat": lat, "lon": lon}
