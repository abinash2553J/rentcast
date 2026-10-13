FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY api ./api
COPY app ./app
COPY models ./models
ENV PYTHONPATH=/srv/src
EXPOSE 8000 8501
# default: API. For the UI: docker run -p 8501:8501 <img> streamlit run app/streamlit_app.py --server.address=0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
