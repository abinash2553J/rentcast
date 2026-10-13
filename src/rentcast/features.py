"""Feature engineering. The SAME function builds features for training and serving, so the two cannot drift."""
from __future__ import annotations

import numpy as np
import pandas as pd


def slug(amenity: str) -> str:
    return "am_" + amenity.strip().lower().replace("/", "_").replace(" ", "_")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw listing columns -> model features. Needs: square_feet, bedrooms, bathrooms, latitude, longitude,
    state, amenities (comma-separated string or NaN), pets_allowed (string or NaN)."""
    sqft = df.square_feet.astype(float)
    X = pd.DataFrame(index=df.index)
    X["sqft"] = sqft
    X["log_sqft"] = np.log(sqft)
    X["bedrooms"] = df.bedrooms.fillna(1).astype(float)
    X["bathrooms"] = df.bathrooms.fillna(1).astype(float)
    X["sqft_per_bed"] = sqft / (X.bedrooms + 1)
    X["lat"] = df.latitude.astype(float)
    X["lon"] = df.longitude.astype(float)
    pets = df.pets_allowed.fillna("")
    X["pets_cats"] = pets.str.contains("Cats").astype(int)
    X["pets_dogs"] = pets.str.contains("Dogs").astype(int)

    am = df.amenities.fillna("").str.get_dummies(sep=",")
    am = am.loc[:, [c for c in am.columns if c.strip()]]
    am.columns = [slug(c) for c in am.columns]
    X = pd.concat([X, am], axis=1)
    X["n_amenities"] = am.sum(axis=1)
    X = pd.concat([X, pd.get_dummies(df.state, prefix="state").astype(int)], axis=1)
    return X
