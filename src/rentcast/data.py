"""Loading and cleaning the UCI "Apartment for Rent Classified" dataset."""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def load_raw(path: str) -> pd.DataFrame:
    """The CSV is semicolon-separated, cp1252-encoded, and uses the literal string "null" for missing."""
    return pd.read_csv(path, sep=";", encoding="cp1252", na_values=["null", "None"], low_memory=False)


def _step(name: str, before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    log.info("cleaning: %-38s removed %6d rows", name, len(before) - len(after))
    return after


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, non-apartment / non-monthly rows, placeholders and outliers; logs rows removed per step."""
    df = _step("duplicate listings", df, df.drop_duplicates(subset=["title", "body", "price"]))
    df = _step("not a monthly apartment listing", df,
               df[df.category.eq("housing/rent/apartment") & df.price_type.eq("Monthly")])
    df = _step("missing price/size/location", df,
               df.dropna(subset=["price", "square_feet", "latitude", "longitude", "state"]))
    # tiny sqft values are placeholders; a few listings have 50,000 sqft or $52,500 typos
    df = _step("implausible sqft or price", df,
               df[df.square_feet.between(200, 5000) & df.price.between(300, 15000)])
    ppsf = df.price / df.square_feet
    df = _step("extreme price per sqft (outer 1%)", df, df[ppsf.between(ppsf.quantile(0.005), ppsf.quantile(0.995))])
    return df.reset_index(drop=True)
