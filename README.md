# Machine Learning Model for Yield Prediction

Predict **grain yield (kg/ha)** from remote-sensing vegetation indices using a suite of
machine learning models — Linear Regression, Random Forest, Gradient Boosting, SVR, and
optional XGBoost — enhanced with **AgroSense**, a RAG (Retrieval-Augmented Generation)
agent that also searches the live internet for context and answers.

## Features

- **Index vs yield analysis** — ranks single vegetation indices and searches the best
  index combinations for predicting yield.
- **Multi-model training** — evaluates Linear Regression, Random Forest, Gradient
  Boosting, SVR, and XGBoost (when installed), including 20-fold cross-validation.
- **Overfitting guard** — selects the best model/features by a constraint on the
  train–test R² gap (`MAX_TRAIN_TEST_DIFF`).
- **Rich outputs** — Excel workbooks, PNG charts, predictions/residuals/metrics for
  both training and test sets, and a zipped results archive.
- **MLflow tracking** — auto-logs each run's params and metrics so dates/runs can be
  compared over time.
- **Docker + CI** — reproducible container runtime and automatic GitHub Actions
  verification on every push.
- **AgroSense RAG agent** — generates a natural-language report and optional live
  interactive chat about the ML results.

## Files

| File | Purpose |
|------|---------|
| `Latest Updated Code for IDLE.py` | Main ML analysis script (RAG agent hooks in here). |
| `rag_agent.py` | **AgroSense** — standalone RAG agent (ingestion + retrieval + generation + live web search). |
| `Bands&VI data_ML.xlsx` | Input dataset (bands + vegetation indices + measured yield). |
| `requirements.txt` | Python dependencies. |
| `.env.example` | RAG provider / API-key template (copy to `.env`). |
| `.gitignore` | Excludes secrets (`.env`) and generated outputs. |
| `Dockerfile` / `docker-compose.yml` | Containerized reproducible runtime. |
| `run.bat` / `run.sh` | One-click launchers (Windows / Unix). |
| `.github/workflows/ci.yml` | GitHub Actions CI (auto-run + verification). |
| `Latest Code - Linear Regression.py` etc. | Per-model variant scripts. |

## How it works

1. **Load & prep** — reads the `2024-25` sheet, cleans column names, converts dates,
   and filters to the requested date.
2. **Indices** — builds the `logM` index and exposes all numerical bands/indices as
   candidate features (excluding yield/label columns).
3. **Single-index regression** — fits Linear Regression per index and ranks by R².
4. **Combination search** — evaluates all feature combinations up to `MAX_COMBINATION_SIZE`
   and ranks by R².
5. **Model training** — for each ML model, picks the best combination under the
   train/test gap constraint, with cross-validation.
6. **Reporting** — writes Excel sheets, PNG charts, and (optionally) the AgroSense
   report, then zips the per-date results.

### Outputs

Per date (under `Datewise_Results/<date>/`):

- `*.xlsx` — single indices, best combinations, ML model results, training/test
  predictions, residuals, and metrics.
- `*.png` — top indices, correlation heatmap, best combinations, actual-vs-predicted,
  training residuals, and feature importance.
- `<date>_rag_report.txt` — AgroSense natural-language report.
- `<date>_Results.zip` — archived results folder.

## AgroSense — the RAG agent

`rag_agent.py` (named **AgroSense**) is a self-contained module that:

1. **Ingests** the ML results (best model, CV/train/test R², best features, top
   indices, best combinations) plus domain knowledge about remote-sensing indices.
2. **Retrieves** relevant context — from BOTH the local knowledge base AND the
   **live internet** (Tavily web search, keyless by default).
3. **Generates** natural-language answers via a pluggable LLM backend, citing the
   retrieved local + web context.

### LLM providers (pluggable, select with `RAG_PROVIDER`)

| Provider | Env var          | Requires API key? | Notes                          |
|----------|------------------|-------------------|--------------------------------|
| `local`  | —                | No                | Heuristic answers + live web   |
| `gemini` | `GOOGLE_API_KEY` | Yes               | `pip install google-generativeai` |
| `openai` | `OPENAI_API_KEY` | Yes               | `pip install openai`           |
| `auto`   | —                | Optional          | Picks gemini → openai → local  |

```env
# .env  (auto-loaded by the scripts)
RAG_PROVIDER=auto
GOOGLE_API_KEY=your_gemini_key
OPENAI_API_KEY=sk-...
```

### Live web search

AgroSense searches the web via **Tavily**. It works **keyless out of the box**
(official `X-Tavily-Access-Mode: keyless` mode, rate-limited). For higher limits, add
a free Tavily key in `.env`:

```env
TAVILY_API_KEY=tvly-...
```

Alternative keyed backends (Brave / SerpAPI) are also supported if `TAVILY_API_KEY`
is absent.

### Connecting the RAG agent to the ML code

At the end of `Latest Updated Code for IDLE.py`, after models are trained and Excel
outputs are written, the script:

1. Builds a context dict of the ML results (`build_rag_context`).
2. Creates a `RAGAgent` (AgroSense) and ingests the results.
3. Prints an **ML performance recommendation** (best model + features).
4. Writes an enriched natural-language report (local + live web context) to
   `Datewise_Results/<date>_rag_report.txt`.
5. Optionally runs an interactive chat (set `RAG_INTERACTIVE = True`).

> The RAG agent is wrapped in a `try/except`, so if it fails (or is disabled via
> `RAG_USE_AGENT = False`), the ML analysis completes normally.

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) configure the RAG provider
copy .env.example .env   # then edit with your API keys

# 3. Run the analysis
python "Latest Updated Code for IDLE.py"
```

Place `Bands&VI data_ML.xlsx` in the same directory, enter a date when prompted
(e.g. `08-Mar`), and the script runs the full analysis + AgroSense report.

You can also pass the date as a command-line argument for non-interactive runs:

```bash
python "Latest Updated Code for IDLE.py" 08-Mar
```

### One-click launchers

For convenience, use the provided launcher scripts (same folder as the `.xlsx`):

**Windows** — double-click `run.bat` (prompts for a date), or from a terminal:

```bat
run.bat
run.bat 08-Mar
run.bat 07-Feb --smoke
```

**macOS / Linux** (make it executable first: `chmod +x run.sh`):

```bash
./run.sh
./run.sh 08-Mar
./run.sh 07-Feb --smoke
```

## Configuration

Key settings at the top of `Latest Updated Code for IDLE.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `TARGET_COL` | `GY (kg/ha)` | Yield column to predict. |
| `MAX_COMBINATION_SIZE` | `3` | Max number of indices per combination. |
| `TEST_SIZE` | `0.2` | Held-out test fraction. |
| `CV_SPLITS` | `20` | Number of cross-validation splits. |
| `MAX_TRAIN_TEST_DIFF` | `0.10` | Max acceptable train–test R² gap (overfitting guard). |
| `RAG_USE_AGENT` | `True` | Enable the AgroSense agent. |
| `RAG_PROVIDER` | `auto` | LLM backend for the agent. |

## MLflow experiment tracking

Each run is auto-logged to MLflow when `mlflow` is installed (script disables it
gracefully if not). Parameters (date, model, features, combos, splits) and metrics
(CV / train / test R², train-test gap) for every date are recorded under `./mlruns`
so you can compare runs over time.

```bash
# Run a full analysis (records an MLflow run)
python "Latest Updated Code for IDLE.py" 08-Mar

# View the tracking UI
mlflow ui
# -> open http://localhost:5000
```

To disable MLflow: `MLFLOW_ENABLED=0 python ...`. To point at a remote server:
`MLFLOW_TRACKING_URI=... python ...`.

> `mlruns/` is gitignored — experiments are local unless you configure a server.

## Docker (reproducible runtime)

Build and run the analysis in a container so you never depend on local package
versions:

```bash
docker build -t ml-yield-prediction .

# Full run for a date (results written to ./out)
docker run --rm -v "$(pwd)/out:/app/out" ml-yield-prediction 08-Mar

# Fast smoke test
docker run --rm -v "$(pwd)/out:/app/out" ml-yield-prediction 08-Mar --smoke

# Or with Docker Compose (smoke test by default)
docker compose run --rm mleval 08-Mar --smoke
```

For interactive RAG chat or MLflow UI, run the script directly with Python instead
of inside the container.

## CI / CD (GitHub Actions)

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to
`main`:

1. Installs dependencies on a fresh runner.
2. Verifies the data file and main script are present.
3. Runs the full pipeline in fast **smoke mode** (`08-Mar --smoke`) to confirm it
   completes without errors — catching regressions such as the runtime failures this
   repo originally had.
4. Uploads the generated results as a downloadable **artifact** (kept 7 days).

This means every commit is automatically verified to produce output, which is a good
sanity check before running the expensive full analysis locally.
