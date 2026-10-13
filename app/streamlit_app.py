"""Streamlit UI: `streamlit run app/streamlit_app.py`"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from rentcast.predict import RentModel  # noqa: E402

st.set_page_config(page_title="rentcast", page_icon="🏙️", layout="centered")


@st.cache_resource
def get_model() -> RentModel:
    return RentModel.load(ROOT / "models" / "rent_model.joblib")


model = get_model()
st.title("🏙️ rentcast")
st.caption("Estimate monthly rent for a US apartment from its size, location and amenities.")

states = sorted(model.cities)
c1, c2 = st.columns(2)
state = c1.selectbox("State", states, index=states.index("TX") if "TX" in states else 0)
city = c2.selectbox("City", sorted(model.cities[state]))
c3, c4, c5 = st.columns(3)
sqft = c3.number_input("Size (sq ft)", 200, 5000, 900, step=50)
beds = c4.selectbox("Bedrooms", [0, 1, 2, 3, 4, 5], index=2, format_func=lambda b: "Studio" if b == 0 else str(b))
baths = c5.selectbox("Bathrooms", [1, 1.5, 2, 2.5, 3, 4], index=0)
pets = st.selectbox("Pets", ["None", "Cats", "Dogs", "Cats,Dogs"])
amenities = st.multiselect("Amenities", model.amenities, default=[a for a in ("Parking", "Gym") if a in model.amenities])

if st.button("Estimate rent", type="primary"):
    res = model.predict(state=state, city=city, square_feet=sqft, bedrooms=beds, bathrooms=baths,
                        amenities=amenities, pets=pets)
    st.metric("Estimated monthly rent", f"${res['mid']:,}")
    st.write(f"**80% range:** ${res['low']:,} – ${res['high']:,}")
    st.map(pd.DataFrame({"lat": [res["lat"]], "lon": [res["lon"]]}), zoom=9)

with st.expander("How accurate is this?"):
    m = model.metrics
    st.write(f"Evaluated on cities the model had never seen: **R² {m['r2']}**, **MAE ${m['mae_usd']:,.0f}** "
             f"(vs ${m['baseline_mae_usd']:,.0f} for always guessing the median). "
             f"The 80% range contained the true rent **{m['interval_80_coverage']:.0%}** of the time. "
             "Data: 2019 listings, ~91% from a single site, so treat estimates as a rough guide.")
