# Machine-Learning-Remote-Sensing-Yield-Prediction-

Yield prediction from remote-sensing vegetation indices using multiple ML models
(Linear Regression, Random Forest, Gradient Boosting, SVR, and optional XGBoost),
enhanced with **AgroSense**, a RAG (Retrieval-Augmented Generation) agent that can
also search the live internet for answers.

## Files

- `Latest Updated Code for IDLE.py` — main ML analysis script (connect the RAG agent here).
- `rag_agent.py` — **AgroSense**: standalone RAG agent (knowledge ingestion + retrieval + generation + live web search).
- `Latest Code - Linear Regression.py`, `... - Random Forest.py`, `... - SVR.py`,
  `... - XGBoost.py`, `Grad Boost updatee.py`, `full Code for collab.py` — per-model variants.
- `requirements.txt` — dependencies.
- `.env.example` — RAG provider / API-key configuration template (copy to `.env`).
- `.gitignore` — excludes secrets (`.env`) and generated outputs.

## AgroSense — the RAG agent

`rag_agent.py` (named **AgroSense**) is a self-contained module that:

1. **Ingests** the ML results (best model, CV/train/test R2, best features, top
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

### Live web search (find solutions from the internet)

AgroSense searches the web via **Tavily**. It works **keyless out of the box**
(official `X-Tavily-Access-Mode: keyless` mode, Search/Extract, rate-limited). For
higher limits, add a free Tavily key in `.env`:

```env
TAVILY_API_KEY=tvly-...
```

Alternative keyed backends (Brave / SerpAPI) are also supported if `TAVILY_API_KEY`
is absent.

## How the RAG agent connects to the ML code

At the end of `Latest Updated Code for IDLE.py`, after models are trained and the
Excel outputs are written, the script:

1. Builds a context dict of the ML results (`build_rag_context`).
2. Creates a `RAGAgent` (AgroSense) and ingests the results.
3. Prints an **ML performance recommendation** (best model + features).
4. Writes an enriched natural-language report (local + live web context) to
   `Datewise_Results/<date>_rag_report.txt`.
5. Optionally runs an interactive chat (set `RAG_INTERACTIVE = True` in the script).

> The RAG agent is wrapped in a `try/except`, so if it fails (or is disabled via
> `RAG_USE_AGENT = False`), the ML analysis completes normally.

## Usage

```bash
pip install -r requirements.txt
python "Latest Updated Code for IDLE.py"
```

Place `Bands&VI data_ML.xlsx` in the same directory, enter a date (e.g. `08-Mar`),
and the script will run the full analysis + AgroSense report (local + live web).

