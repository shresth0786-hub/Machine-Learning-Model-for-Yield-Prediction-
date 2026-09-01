# =============================================================================
# ML Yield Prediction — reproducible runtime
# =============================================================================
FROM python:3.11-slim

# Keep image lean
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY "Latest Updated Code for IDLE.py" .
COPY rag_agent.py .
COPY "Bands&VI data_ML.xlsx" .

# Run the analysis with a dummy/arg date. Override with the DATE argument and
# --smoke flag as needed:
#   docker run --rm -v "$(pwd)/out:/app/out" <image> 08-Mar
ENTRYPOINT ["python", "Latest Updated Code for IDLE.py"]
