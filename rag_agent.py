# =============================================================================
# AGROSENSE RAG AGENT
# Retrieval-Augmented Generation for Yield Prediction Analysis
# =============================================================================
# A self-contained RAG agent (named "AgroSense") that:
#  1. INGESTS the ML results produced by the main analysis script into a
#     retrievable knowledge store (chunks + metadata).
#  2. RETRIEVES the most relevant context for a question - from BOTH the
#     local knowledge base AND the live internet (web search).
#  3. GENERATES natural-language answers using a pluggable LLM backend.
#
# Supported backends (pluggable by env var RAG_PROVIDER):
#   - "gemini"  -> Google Generative AI (needs GOOGLE_API_KEY)      [default if key set]
#   - "openai"  -> OpenAI ChatGPT (needs OPENAI_API_KEY)
#   - "local"   -> heuristic rule-based answerer (NO API KEY REQUIRED)
#
# WEB SEARCH: if no LLM key is configured, AgroSense falls back to live web
# search results so it can still find answers/solutions for questions that are
# not covered by the local knowledge base.
#
# The agent also exposes an "agent_decision" helper that uses the retrieved
# ML results to recommend the best model / features so the workflow picks
# high-performing configurations reliably (helps the ML models perform well).
# =============================================================================

import os
import sys
import json
import re
import difflib
import urllib.parse

import numpy as np
import pandas as pd

from html.parser import HTMLParser

# Console-safe printing: web content may contain Unicode that the Windows
# cp1252 console cannot encode, which would crash print(). Reconfigure stdout
# to UTF-8 and replace unencodable characters instead of raising.
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ------------------------- Agent identity -----------------------------------
AGENT_NAME = "AgroSense"
AGENT_TAGLINE = (
    "AgroSense - your remote-sensing yield intelligence agent. "
    "Answers from local ML results + live web knowledge."
)

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False


# -----------------------------------------------------------------------------
# LIGHTWEIGHT .env LOADER (no external dependency)
# Loads KEY=VALUE lines from .env / .env.local into os.environ if not set.
# This is where TAVILY_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY live.
# -----------------------------------------------------------------------------
def _load_dotenv():
    for fname in (".env.local", ".env"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            continue


_load_dotenv()

# -----------------------------------------------------------------------------
# OPTIONAL LLM BACKENDS (only imported if actually used)
# -----------------------------------------------------------------------------
def _has(module):
    try:
        __import__(module)
        return True
    except Exception:
        return False


# =============================================================================
# WEB SEARCH (live internet retrieval - no API key required)
# Uses DuckDuckGo HTML search and fetches the top result pages as context.
# =============================================================================

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


class _ResultParser(HTMLParser):
    """Extract text from a simple HTML page (for reading search result bodies)."""

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip == 0:
            self.parts.append(data)


def _clean_text(html):
    p = _ResultParser()
    try:
        p.feed(str(html))
    except Exception:
        return ""
    text = " ".join(" ".join(p.parts).split())
    return text


# -----------------------------------------------------------------------------
# Web search backend selection
#
# Search providers are tried in this order (first that works wins):
#   1. Tavily   (TAVILY_API_KEY)   - RAG-friendly, free tier
#   2. Brave    (BRAVE_API_KEY)    - free tier
#   3. SerpAPI  (SERP_API_KEY)     - free tier
#   4. DuckDuckGo (NO KEY, best-effort - may be rate-limited/bot-blocked)
# -----------------------------------------------------------------------------

def _tavily_search(query, max_results=4):
    """Tavily search. Uses TAVILY_API_KEY when set, otherwise keyless mode.

    Keyless mode (official, no account): send header
    X-Tavily-Access-Mode: keyless and omit the API key. Search/Extract only,
    rate-limited. This lets the agent find live answers with zero setup.
    """
    key = os.environ.get("TAVILY_API_KEY")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    else:
        headers["X-Tavily-Access-Mode"] = "keyless"
    payload = {"query": query, "max_results": max_results,
               "search_depth": "basic", "include_answer": True}
    r = requests.post("https://api.tavily.com/search", json=payload,
                      headers=headers, timeout=25)
    if r.status_code == 401:
        raise RuntimeError("Tavily unauthorized (no/expired API key)")
    if r.status_code == 432:
        raise RuntimeError("Tavily plan limit reached")
    r.raise_for_status()
    data = r.json()
    out = []
    for item in data.get("results", []):
        out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                    "snippet": item.get("content", ""), "body": item.get("content", "")})
    if data.get("answer"):
        out.insert(0, {"title": "Tavily AI Answer", "url": "", "snippet": data["answer"]})
    return out[:max_results]


def _brave_search(query, max_results=4):
    key = os.environ.get("BRAVE_API_KEY")
    if not key:
        raise RuntimeError("no BRAVE_API_KEY")
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     params={"q": query, "count": max_results},
                     headers={"X-Subscription-Token": key, "Accept": "application/json"},
                     timeout=25)
    r.raise_for_status()
    out = []
    for item in r.json().get("web", {}).get("results", []):
        out.append({"title": item.get("title", ""), "url": item.get("url", ""),
                    "snippet": item.get("description", "")})
    return out


def _serpapi_search(query, max_results=4):
    key = os.environ.get("SERP_API_KEY")
    if not key:
        raise RuntimeError("no SERP_API_KEY")
    r = requests.get("https://serpapi.com/search.json",
                     params={"engine": "google", "q": query, "api_key": key, "num": max_results},
                     timeout=25)
    r.raise_for_status()
    out = []
    for item in r.json().get("organic_results", []):
        out.append({"title": item.get("title", ""), "url": item.get("link", ""),
                    "snippet": item.get("snippet", "")})
    return out


def _ddg_search(query, max_results=4):
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query, "ia": "web", "kp": "-2"},
        headers={"User-Agent": _UA},
        timeout=20,
    )
    if resp.status_code != 200:
        raise RuntimeError("DuckDuckGo blocked")
    try:
        items = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
            r'.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            resp.text,
            re.S,
        )
    except Exception:
        items = []
    seen, out = set(), []
    for url, title, snippet in items:
        url = _normalize_ddg_url(url)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({"title": _clean_text(title) or url, "url": url,
                    "snippet": _clean_text(snippet)})
        if len(out) >= max_results:
            break
    if not out:
        raise RuntimeError("DuckDuckGo returned no parseable results")
    return out


def web_search(query, max_results=4, num_snippets=8):
    """Search the live web (multi-provider) and return a list of result dicts.

    Returns results with keys: title, url, snippet (and optional body).
    Returns [] on total failure.
    """
    if not REQUESTS_AVAILABLE:
        return []
    providers = [
        ("Tavily", _tavily_search),
        ("Brave", _brave_search),
        ("SerpAPI", _serpapi_search),
        ("DuckDuckGo", _ddg_search),
    ]
    for name, fn in providers:
        try:
            results = fn(query, max_results=max_results)
            if results:
                return results
        except Exception:
            continue
    return []


def _normalize_ddg_url(url):
    """Convert DuckDuckGo redirect URLs to real destination URLs."""
    if url.startswith("//"):
        url = "https:" + url
    if "duckduckgo.com/l/?uddg=" in url:
        m = re.search(r"uddg=([^&]+)", url)
        if m:
            try:
                from urllib.parse import unquote
                return unquote(m.group(1))
            except Exception:
                return url
    return url


def build_web_context(query, max_results=3):
    """Return a formatted text block of live web search results."""
    res = web_search(query, max_results=max_results)
    if not res:
        return ""
    lines = ["--- LIVE WEB RESULTS ---"]
    for i, r in enumerate(res, 1):
        lines.append(f"{i}. {r.get('title','')}")
        lines.append(f"   URL: {r.get('url','')}")
        lines.append(f"   {r.get('snippet','')}")
        if r.get("body"):
            lines.append(f"   (page excerpt) {r['body'][:800]}")
    lines.append("--- END WEB RESULTS ---")
    return "\n".join(lines)


# =============================================================================
# 1) KNOWLEDGE BASE (static domain knowledge about remote-sensing indices)
# =============================================================================

INDEX_KNOWLEDGE = {
    "ndvi": (
        "NDVI (Normalized Difference Vegetation Index) measures vegetation "
        "greenness/health using (NIR - Red)/(NIR + Red). High NDVI indicates "
        "dense, healthy vegetation. It is a strong predictor of crop biomass "
        "and photosynthetic activity, hence a reliable proxy for yield."
    ),
    "ndwi": (
        "NDWI (Normalized Difference Water Index) highlights water content / "
        "moisture stress in vegetation using green and NIR bands. Useful for "
        "irrigation management and detecting drought stress that lowers yield."
    ),
    "gndvi": (
        "GNDVI (Green NDVI) uses the green band instead of red: "
        "(NIR - Green)/(NIR + Green). More sensitive to chlorophyll content "
        "and often saturates later than NDVI, making it useful in dense canopy."
    ),
    "savi": (
        "SAVI (Soil Adjusted Vegetation Index) corrects NDVI for soil "
        "background influence using a soil adjustment factor L. Useful in "
        "early growth / partial canopy where soil is visible."
    ),
    "lai": (
        "LAI (Leaf Area Index) relates to the area of leaves per ground area. "
        "Drives light interception and biomass accumulation, strongly "
        "correlated with final grain yield."
    ),
    "ci": (
        "CI (Chlorophyll Index) reflects chlorophyll concentration in the "
        "canopy, linked to nitrogen status and photosynthetic capacity, "
        "influencing grain filling and yield."
    ),
    "reb": (
        "Red-edge based indices are sensitive to chlorophyll and are valuable "
        "for monitoring crop nitrogen and health without saturation in dense "
        "vegetation."
    ),
    "logm": (
        "logM is a custom log-transform composite index built from NIR, Green, "
        "Red-edge minus Red and Blue bands. Logarithmic scaling compresses "
        "dynamic range and can linearise the yield relationship, often "
        "improving regression performance on skewed spectral data."
    ),
    "nirs": (
        "Near-infrared reflectance responds to cell structure and canopy "
        "density. High NIR generally indicates vigorous vegetation and higher "
        "potential yield."
    ),
    "gndvi2": (
        "Green band reflectance is sensitive to chlorophyll and plant "
        "physiology; combining green with NIR improves sensitivity to "
        "crop health during peak season."
    ),
}

GENERAL_KNOWLEDGE = [
    (
        "Yield prediction with remote sensing",
        "Combining multiple vegetation indices usually outperforms single "
        "indices because different indices capture complementary crop "
        "physiology (greenness, moisture, chlorophyll, biomass). Linear "
        "regression gives interpretable coefficients; tree-based models "
        "(Random Forest, Gradient Boosting, XGBoost) capture non-linear "
        "interactions but need enough samples to avoid overfitting."
    ),
    (
        "Model comparison and overfitting",
        "Compare models using cross-validated R2 (CV R2) rather than training "
        "R2. A large gap between training and test R2 signals overfitting. "
        "Use RMSE and MAE to judge practical prediction error in kg/ha. "
        "Small sample sizes favour simpler models (Linear Regression, tuned "
        "Gradient Boosting) over very flexible ones that overfit."
    ),
    (
        "How to make the ML models perform well",
        "1) Feed the model the best-performing combination of indices rather "
        "than all features. 2) Use cross-validation to select the model. "
        "3) Keep the train-test gap small to avoid overfitting. 4) Standardize "
        "features for SVR. 5) Use the test R2 + RMSE together to judge quality. "
        "6) Combine complementary indices (greenness + moisture + chlorophyll)."
    ),
]


# =============================================================================
# 2) KNOWLEDGE INGESTION - build chunks from ML results + domain knowledge
# =============================================================================

class KnowledgeManager:
    """Holds chunked knowledge and performs simple retrieval."""

    def __init__(self):
        self.chunks = []          # list of dicts: {"content": str, "tags": [..]}
        self.context = {}         # raw ML context dict passed by main script

    # -------------------------------------------------------------------------
    def add_domain_knowledge(self):
        for name, text in INDEX_KNOWLEDGE.items():
            self.chunks.append({"content": f"{name.upper()} index: {text}", "tags": [name]})
        for title, text in GENERAL_KNOWLEDGE:
            self.chunks.append({"content": f"{title}: {text}", "tags": [t.lower() for t in title.split()]})

    # -------------------------------------------------------------------------
    def add_ml_results(self, ctx):
        """Turn the ML results dict into retrievable chunks."""
        self.context = ctx or {}
        c = self.context

        if c.get("best_model"):
            self.chunks.append({
                "content": (
                    f"Best model: {c.get('best_model')}. CV R2 = {c.get('best_cv_r2')}, "
                    f"Train R2 = {c.get('best_train_r2')}, Test R2 = {c.get('best_test_r2')}, "
                    f"gap = {c.get('best_gap')}. Best features: {c.get('best_features')}."
                ),
                "tags": ["best", "model", str(c.get('best_model', '')).lower()],
            })

        models = c.get("models") or []
        if models:
            lines = ["; ".join(f"{m['Model']}: R2={m.get('R2_Score')}" for m in models)]
            self.chunks.append({"content": "Model results: " + lines[0], "tags": ["model", "results"]})

        top_indices = c.get("top_indices") or []
        if top_indices:
            txt = "Top indices by R2: " + "; ".join(
                f"{r.get('Index')}(R2={r.get('R2_Score')}, RMSE={r.get('RMSE')})" for r in top_indices
            )
            self.chunks.append({"content": txt, "tags": ["index", "top"]})

        combos = c.get("combinations") or []
        if combos:
            txt = "Best index combinations: " + "; ".join(
                f"{r.get('Combination')}(R2={r.get('R2_Score')})" for r in combos[:5]
            )
            self.chunks.append({"content": txt, "tags": ["combination"]})

        if c.get("n_samples"):
            self.chunks.append({
                "content": f"Dataset: {c.get('n_samples')} samples for {c.get('date')}.",
                "tags": ["data", "samples"],
            })

    # -------------------------------------------------------------------------
    def retrieve(self, question, k=3):
        """Simple keyword/embedding-free retrieval scoring by token overlap."""
        tokens = {t.lower() for t in question.replace("?", "").split() if len(t) > 2}
        scored = []
        for ch in self.chunks:
            score = 0
            text = ch["content"].lower()
            for tok in tokens:
                if tok in text:
                    score += 1
            for tag in ch.get("tags", []):
                if tag in question.lower():
                    score += 2
            # match against all raw context field names too
            for key in self.context:
                if key.lower().replace("_", " ") in question.lower():
                    score += 1
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ch for s, ch in scored[:k]]

    def knowledge_context(self, question):
        retrieved = self.retrieve(question)
        return "\n".join(ch["content"] for ch in retrieved)


# =============================================================================
# 3) GENERATION BACKENDS (pluggable LLMs + local fallback)
# =============================================================================

def _gemini_generate(prompt, max_tokens=900):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens})
    return resp.text


def _openai_generate(prompt, max_tokens=900):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise agricultural remote-sensing yield analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _local_generate(prompt, retrieved, web_ctx=""):
    """Heuristic answerer - no API key needed. Safe as default fallback."""
    lines = [
        f"=== {AGENT_NAME.upper()} (LOCAL MODE) ===",
        "",
        "Retrieved local context:",
    ]
    for ch in retrieved:
        lines.append("  - " + ch["content"])
    if web_ctx:
        lines.append("")
        lines.append("Live web answers found:")
        lines.append(web_ctx)
    lines.append("")
    lines.append("Answer (heuristic, no LLM):")
    lines.append(
        "Based on the retrieved ML results, the model with the highest "
        "cross-validated R2 and a small train-test gap should be preferred. "
        "Use its best feature combination for new predictions. For richer "
        "natural-language insights, set an LLM API key "
        "(GOOGLE_API_KEY / OPENAI_API_KEY)."
    )
    return "\n".join(lines)


# =============================================================================
# 4) MAIN RAG AGENT
# =============================================================================

class RAGAgent:
    """Retrieval-augmented agent for yield-prediction ML results."""

    def __init__(self, provider=None, api_key=None):
        # Auto-select the provider if not forced:
        #   explicit RAG_PROVIDER -> that; else gemini if key; else openai if
        #   key; else local.
        env_prov = (os.environ.get("RAG_PROVIDER") or "auto").lower()
        if provider in ("gemini", "openai", "local"):
            self.provider = provider
        elif api_key:
            self.provider = "gemini" if (os.environ.get("RAG_PROVIDER", "gemini") == "gemini") else "openai"
        elif env_prov in ("gemini", "openai", "local"):
            self.provider = env_prov
        elif os.environ.get("GOOGLE_API_KEY"):
            self.provider = "gemini"
        elif os.environ.get("OPENAI_API_KEY"):
            self.provider = "openai"
        else:
            self.provider = "local"

        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key if self.provider == "gemini" else os.environ.get("GOOGLE_API_KEY", "x")
            os.environ["OPENAI_API_KEY"] = api_key if self.provider == "openai" else os.environ.get("OPENAI_API_KEY", "x")

        self.km = KnowledgeManager()
        self.km.add_domain_knowledge()

    # -------------------------------------------------------------------------
    def ingest(self, context_dict):
        """Call after ML analysis with a dict of results."""
        self.km.add_ml_results(context_dict)
        return len(self.km.chunks)

    # -------------------------------------------------------------------------
    def _generate(self, question, retrieved, web_ctx="", system_guidance=""):
        context = "\n".join(ch["content"] for ch in retrieved)
        prompt = (
            f"You are {AGENT_NAME}, an agricultural remote-sensing yield "
            f"analyst. Answer accurately and helpfully.\n"
            f"{system_guidance}\n\n"
            f"Use the retrieved context below as your primary evidence. "
            f"You may also use the live web results to enrich or update the "
            f"answer. Cite the source URL when you rely on web info.\n"
            f"--- RETRIEVED CONTEXT (local ML results + domain) ---\n{context}\n"
            f"--- END RETRIEVED CONTEXT ---\n"
            f"{web_ctx}\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        try:
            if self.provider == "gemini" and _has("google.generativeai") and os.environ.get("GOOGLE_API_KEY"):
                return _gemini_generate(prompt)
            if self.provider == "openai" and _has("openai") and os.environ.get("OPENAI_API_KEY"):
                return _openai_generate(prompt)
        except Exception as e:
            return (f"[INFO] LLM backend errored ({e}). Falling back to local mode.\n\n"
                    + _local_generate(prompt, retrieved, web_ctx))
        # default fallback
        return _local_generate(prompt, retrieved, web_ctx)

    # -------------------------------------------------------------------------
    def answer(self, question, use_web=True):
        """Retrieve relevant local knowledge (+ live web) and generate an answer."""
        retrieved = self.km.retrieve(question, k=3)
        web_ctx = ""
        if use_web and REQUESTS_AVAILABLE:
            web_ctx = build_web_context(question, max_results=3)
        return self._generate(question, retrieved, web_ctx)

    # -------------------------------------------------------------------------
    def agent_decision(self):
        """Return recommended model + features and diagnostics (helps ML quality)."""
        c = self.km.context
        recommendations = []
        if c.get("best_model"):
            recommendations.append(f"Use model: {c['best_model']}")
        if c.get("best_features"):
            recommendations.append(f"with features: {c['best_features']}")
        if c.get("best_gap") is not None and c.get("best_gap", 1) > 0.20:
            recommendations.append("Caution: train-test gap is large; consider simpler model or more samples.")
        if not recommendations:
            recommendations.append("No ML results ingested yet - run the analysis first.")
        return recommendations

    # -------------------------------------------------------------------------
    def generate_report(self):
        """Produce an end-to-end natural language summary report."""
        c = self.km.context
        headline = (
            f"Best model {c.get('best_model')} achieved CV R2 {c.get('best_cv_r2')}, "
            f"Training R2 {c.get('best_train_r2')}, Testing R2 {c.get('best_test_r2')} "
            f"with features {c.get('best_features')}."
        )
        q = "What is the best model and which indices should I use for yield prediction?"
        body = self.answer(q, use_web=True)
        return f"{AGENT_NAME} REPORT\n{'='*40}\n\n{body}\n\nSummary:\n{headline}"


# =============================================================================
# 5) INTERACTIVE CHAT LOOP (used by main script)
# =============================================================================

def run_interactive_chat(agent):
    print("\n" + "=" * 90)
    print(f" {AGENT_NAME} - ASK QUESTIONS ABOUT YOUR YIELD ANALYSIS (web search ON)")
    print(f" Provider: {agent.provider.upper()}  (type 'exit' to quit)")
    print("=" * 90)
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in {"exit", "quit", "q"}:
            break
        print(f"\n{AGENT_NAME}:\n" + agent.answer(q, use_web=True) + "\n")


# =============================================================================
# 6) CONFIGURATION HELP
# =============================================================================

def setup_hint(agent_provider):
    if agent_provider == "gemini":
        return ("Set GOOGLE_API_KEY env var (https://aistudio.google.com/apikey); "
                "google-generativeai installed.")
    if agent_provider == "openai":
        return "Set OPENAI_API_KEY env var and pip install openai"
    return ("Local mode - no API key required; uses live web search for answers. "
            "Set GOOGLE_API_KEY or OPENAI_API_KEY for richer LLM answers.")


if __name__ == "__main__":
    agent = RAGAgent()
    print(AGENT_TAGLINE)
    print("Auto-selected provider:", agent.provider.upper())
    print("Ingested domain chunks:", len(agent.km.chunks))
    print()
    print(agent.answer("Why does combining multiple vegetation indices improve yield prediction?"))
    print()
    print(agent.answer("What are the latest remote sensing techniques for wheat yield prediction in 2026?", use_web=True))
