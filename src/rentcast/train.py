"""Train median + 10th/90th percentile models on log(rent) and save one bundle.

    python -m rentcast.train --data data/apartments_for_rent_classified_100K.csv --out models/rent_model.joblib
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from .data import clean, load_raw
from .features import build_features, slug

QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}


def _fit(q: float, X: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=400, learning_rate=0.08, random_state=42)
    return m.fit(X, y)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/rent_model.joblib")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = clean(load_raw(args.data))
    X, y = build_features(df), np.log(df.price)
    # Split by city (never random rows): test cities are unseen, so scores reflect real generalisation.
    # A second city-level split carves out a calibration set for the interval widening below.
    groups = df.cityname.fillna("unknown")
    dev, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(X, y, groups))
    tr_i, cal_i = next(GroupShuffleSplit(test_size=0.2, random_state=0).split(X.iloc[dev], y.iloc[dev], groups.iloc[dev]))
    tr, cal = dev[tr_i], dev[cal_i]

    fitted = {k: _fit(q, X.iloc[tr], y.iloc[tr]) for k, q in QUANTILES.items()}
    # Conformalised quantile regression: raw 10-90% quantiles under-cover on unseen cities, so widen them (in
    # log space) by the 80th percentile of how far calibration-city rents fall outside the raw interval.
    lo_c, hi_c = fitted["low"].predict(X.iloc[cal]), fitted["high"].predict(X.iloc[cal])
    scores = np.maximum(lo_c - y.iloc[cal], y.iloc[cal] - hi_c)
    pad = float(max(np.quantile(scores, min(1.0, 0.8 * (1 + 1 / len(scores)))), 0.0))

    lo_t, mid_t, hi_t = (fitted[k].predict(X.iloc[te]) for k in ("low", "mid", "high"))
    true = np.exp(y.iloc[te])
    pred_mid = np.exp(mid_t)
    cover = lambda lo, hi: float(np.mean((true >= np.exp(lo)) & (true <= np.exp(hi))))  # noqa: E731
    metrics = {
        "n_rows_after_cleaning": int(len(df)),
        "r2": round(r2_score(true, pred_mid), 3),
        "mae_usd": round(mean_absolute_error(true, pred_mid), 1),
        "mape": round(float(np.mean(np.abs(pred_mid - true) / true)), 3),
        "baseline_mae_usd": round(mean_absolute_error(true, np.full(len(te), np.exp(y.iloc[tr]).median())), 1),
        "interval_80_coverage_raw": round(cover(lo_t, hi_t), 3),
        "interval_80_coverage": round(cover(lo_t - pad, hi_t + pad), 3),
        "interval_pad_log": round(pad, 3),
    }
    logging.info(json.dumps(metrics, indent=2))

    models = {k: _fit(q, X, y) for k, q in QUANTILES.items()}  # final models use all data
    cities = (df.dropna(subset=["cityname"]).groupby(["state", "cityname"])
              .agg(lat=("latitude", "median"), lon=("longitude", "median"), n=("price", "size")).reset_index())
    cities = cities[cities.n >= 10]
    city_index = {s: {r.cityname: [round(r.lat, 4), round(r.lon, 4)] for r in g.itertuples()}
                  for s, g in cities.groupby("state")}
    raw_amenities = sorted({a.strip() for a in df.amenities.dropna().str.split(",").explode() if a.strip()})
    bundle = {"models": models, "columns": list(X.columns), "cities": city_index, "metrics": metrics, "pad": pad,
              "amenities": [a for a in raw_amenities if slug(a) in X.columns]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    Path(args.out).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
