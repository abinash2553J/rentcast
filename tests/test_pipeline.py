import numpy as np
import pandas as pd
import pytest

from rentcast.data import clean
from rentcast.features import build_features


def raw_rows(n=6, **over):
    base = {
        "title": [f"t{i}" for i in range(n)], "body": [f"b{i}" for i in range(n)],
        "category": ["housing/rent/apartment"] * n, "price_type": ["Monthly"] * n,
        "price": [1000 + 100 * i for i in range(n)], "square_feet": [700 + 50 * i for i in range(n)],
        "latitude": [30.0] * n, "longitude": [-97.0] * n, "state": ["TX"] * n,
        "bedrooms": [1.0] * n, "bathrooms": [1.0] * n,
        "amenities": ["Pool,Gym"] + [None] * (n - 1), "pets_allowed": [None] * n,
    }
    base.update(over)
    return pd.DataFrame(base)


def test_clean_drops_duplicates_and_bad_rows():
    df = raw_rows(n=200, price=[1000 + 5 * i for i in range(200)], square_feet=[800 + i for i in range(200)])
    df = pd.concat([df, df.iloc[[0]]])                    # exact duplicate of row "t0"
    df.loc[1, "price_type"] = "Weekly"                    # non-monthly
    df.loc[2, "square_feet"] = 50000                      # typo outlier
    out = clean(df)
    assert out.title.is_unique                            # duplicate removed
    assert not out.title.isin(["t1", "t2"]).any()         # weekly + outlier removed
    assert out.price.between(300, 15000).all() and out.square_feet.between(200, 5000).all()
    assert len(out) > 180                                 # but the good rows survive


def test_features_multi_hot_amenities_and_state():
    X = build_features(raw_rows())
    assert X.loc[0, "am_pool"] == 1 and X.loc[0, "am_gym"] == 1 and X.loc[0, "n_amenities"] == 2
    assert X.loc[1, "n_amenities"] == 0
    assert "state_TX" in X.columns
    assert not X.isna().any().any()


def test_features_single_row_matches_batch():
    df = raw_rows()
    batch = build_features(df).iloc[[0]]
    single = build_features(df.iloc[[0]].reset_index(drop=True)).reindex(columns=batch.columns, fill_value=0)
    np.testing.assert_allclose(batch.to_numpy(float), single.to_numpy(float))


@pytest.mark.skipif(not __import__("pathlib").Path("models/rent_model.joblib").exists(), reason="model not trained")
def test_model_predictions_are_sane_and_ordered():
    from rentcast.predict import RentModel

    m = RentModel.load("models/rent_model.joblib")
    state = next(iter(m.cities))
    city = next(iter(m.cities[state]))
    small = m.predict(state=state, city=city, square_feet=500, bedrooms=1, bathrooms=1)
    big = m.predict(state=state, city=city, square_feet=1800, bedrooms=3, bathrooms=2, amenities=["Pool", "Gym"])
    assert small["low"] <= small["mid"] <= small["high"]
    assert big["mid"] > small["mid"]
    with pytest.raises(ValueError):
        m.predict(state="ZZ", city="Nowhere", square_feet=800, bedrooms=2, bathrooms=1)
