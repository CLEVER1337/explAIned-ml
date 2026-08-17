# The ranking service (:8002). Deliberately does NOT install the `jobs` group — torch and
# sentence-transformers are ~2.5 GB and this image never encodes anything.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN pip install "poetry>=1.8"

COPY pyproject.toml poetry.lock* ./
COPY packages/feature_store/pyproject.toml ./packages/feature_store/
COPY packages/feature_store/src ./packages/feature_store/src
RUN poetry install --only main --no-root

COPY src ./src
RUN poetry install --only-root

EXPOSE 8002

# Several workers are fine here, unlike the FAISS service: a LightGBM booster is a few
# megabytes, so a second copy costs nothing worth optimising away.
CMD ["uvicorn", "explained_ml.service.app:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]
