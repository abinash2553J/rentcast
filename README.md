# 🏙️ rentcast

Predict monthly US apartment rent from **size, location and amenities** — with a calibrated uncertainty range, a Streamlit UI and a FastAPI service.

Built on the UCI *Apartment for Rent Classified* dataset (~100K listings, Dec 2018 – Dec 2019).

## Results

Evaluated on **cities the model has never seen** (split by city, not by row, so there is no leakage from neighbouring listings).

| | MAE | R² |
|---|---|---|
| Always predict the median rent | $500 | – |
| **rentcast** (gradient boosting, log-price) | **$276** | **0.61** |

The app returns a rent estimate plus an 80% range. Raw quantile models only covered 64% of true rents on unseen cities, so the range is widened with **conformalised quantile regression**; coverage on the held-out cities is now **77%**. The remaining gap is real city-level distribution shift.

## Quickstart

```bash
pip install -r requirements-dev.txt
# put the extracted CSV in data/ (see data/README.md), then:
make train        # cleans data, trains models, writes models/rent_model.joblib
make test
make app          # Streamlit UI  -> http://localhost:8501
make api          # FastAPI       -> http://localhost:8000/docs
```

A trained model is already committed in `models/`, so `make app` works without retraining.

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"state":"TX","city":"Austin","square_feet":900,"bedrooms":2,"bathrooms":1,"amenities":["Pool","Gym"]}'
# {"low":845,"mid":1045,"high":1406,...}
```

## Repo layout

```
src/rentcast/
  data.py       load (semicolon CSV, cp1252, "null" = missing) + logged cleaning steps
  features.py   build_features(): used for BOTH training and serving (no train/serve skew)
  train.py      city-grouped split, quantile GBMs on log(rent), conformal calibration
  predict.py    RentModel: shared by the app and the API
app/streamlit_app.py   UI          api/main.py   FastAPI service
tests/                 cleaning, features, model sanity, API round-trip
Dockerfile · Makefile · .github/workflows/ci.yml
```

## Design decisions worth knowing

- **Log target.** Rents are right-skewed; modelling `log(price)` stabilises errors across cheap and expensive markets.
- **City-grouped split.** A random row split lets the model memorise a city's listings. Splitting by city measures what the app actually does: price a place in a market it may not have seen much of.
- **Features unavailable to a user are excluded** (listing source, posting month, photo count) so the app asks only for things a renter would know.
- **Linear vs tree models.** Ridge reached R² ≈ 0.28 on the same task; trees reach ≈ 0.6 because rent depends nonlinearly on location.

## Limitations

- 2019 data, so absolute prices are stale. The model captures *relative* structure, not today's market.
- ~91% of listings come from one site (RentDigs); coverage is uneven across states and cities (only cities with ≥10 listings are selectable).
- The 80% range is approximate (77% empirical on unseen cities).

## Deploy

- **Streamlit Community Cloud:** push to GitHub → "New app" → main file `app/streamlit_app.py`.
- **Docker (API):** `docker build -t rentcast . && docker run -p 8000:8000 rentcast`
- **Docker (UI):** `docker run -p 8501:8501 rentcast streamlit run app/streamlit_app.py --server.address=0.0.0.0`

## Roadmap

- [ ] Neighbourhood features: out-of-fold median rent of the k nearest listings
- [ ] TF-IDF text features from listing title/body
- [ ] Optuna tuning + LightGBM/CatBoost comparison
- [ ] SHAP explanations in the UI ("why this price?")
- [ ] Error analysis by state and price tier
