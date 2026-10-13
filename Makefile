DATA ?= data/apartments_for_rent_classified_100K.csv

install:
	pip install -r requirements-dev.txt
train:
	PYTHONPATH=src python -m rentcast.train --data $(DATA) --out models/rent_model.joblib
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
api:
	uvicorn api.main:app --reload
