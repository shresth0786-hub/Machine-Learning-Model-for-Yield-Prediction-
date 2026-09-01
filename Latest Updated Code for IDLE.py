# =============================================================================
# ADVANCED MACHINE LEARNING:
# INDEX vs YIELD ANALYSIS
# =============================================================================

import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")

# Console-safe printing: the script prints UTF-8 characters (e.g. checkmarks),
# which can crash on Windows cp1252 consoles. Reconfigure stdout/stderr to
# UTF-8 and replace any unencodable characters instead of raising.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    ShuffleSplit
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.base import clone

# RAG AGENT (retrieval-augmented agent for ML result insights)
# from rag_agent import RAGAgent, run_interactive_chat, setup_hint

# OPTIONAL XGBOOST
try:
    from xgboost import XGBRegressor
    xgb_available = True
except:
    xgb_available = False

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "Bands&VI data_ML.xlsx"

TARGET_COL = "GY (kg/ha)"

OUTPUT_EXCEL = ""

MAX_COMBINATION_SIZE = 3

TEST_SIZE = 0.2

RANDOM_STATE = 42

CV_SPLITS = 20

MAX_TRAIN_TEST_DIFF = 0.10

# -----------------------------------------------------------------------------
# SMOKE TEST MODE
#   When True, the expensive full combination x CV search is reduced to a tiny
#   subset so the analysis finishes in seconds. Used by CI and Docker smoke
#   tests to verify the pipeline end-to-end without waiting many minutes.
#   Enabled via: python script.py <date> --smoke
# -----------------------------------------------------------------------------

SMOKE_TEST = "--smoke" in sys.argv

if SMOKE_TEST:
    CV_SPLITS = 2
    MAX_COMBINATION_SIZE = 2

# -----------------------------------------------------------------------------
# MLFLOW TRACKING
#   When True, each run's params, metrics and (tree) models are logged with
#   MLflow so runs can be compared over time. Run `mlflow ui` to view.
#   Disabled automatically in SMOKE_TEST and in CI unless MLFLOW_TRACKING_URI
#   is explicitly set.
# -----------------------------------------------------------------------------

MLFLOW_ENABLED = os.environ.get("MLFLOW_ENABLED", "1" if not SMOKE_TEST else "0") == "1"

if MLFLOW_ENABLED:
    try:
        import mlflow

        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        )
        mlflow.sklearn.autolog(log_models=False, silent=True)

        _mlflow_run = mlflow.start_run(run_name=os.environ.get(
            "MLFLOW_RUN_NAME",
            f"{os.path.basename(DATA_FILE)}-{os.path.splitext(DATA_FILE)[0]}"
        ))
        mlflow.log_params({
            "cv_splits": CV_SPLITS,
            "test_size": TEST_SIZE,
            "max_combo_size": MAX_COMBINATION_SIZE,
            "random_state": RANDOM_STATE,
            "smoke_test": SMOKE_TEST,
        })
    except Exception as _mle:
        MLFLOW_ENABLED = False
        print(f"[WARN] MLflow disabled: {_mle}")

# -----------------------------------------------------------------------------
# AGROSENSE RAG AGENT SETTINGS
#   RAG_PROVIDER: "auto" | "gemini" | "openai" | "local"
#     - auto   -> uses Gemini if GOOGLE_API_KEY is set, else OpenAI if
#                 OPENAI_API_KEY is set, else local (web-search only).
#     - gemini / openai -> force that LLM backend (needs its API key).
#     - local  -> no API key; answers from local knowledge + live web search.
#   RAG_INTERACTIVE: ask questions after analysis in a chat loop
#   RAG_USE_AGENT: enable the agent integration entirely
# -----------------------------------------------------------------------------

RAG_PROVIDER = os.environ.get("RAG_PROVIDER", "auto")

RAG_INTERACTIVE = False

RAG_USE_AGENT = True

def get_cv_strategy(n_samples, cv_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Return a ShuffleSplit that is safe for small sample sizes.

    Ensures the test set contains at least one sample when `test_size` is
    fractional, and limits the number of splits to at most the number of
    samples (min 2).
    """
    if isinstance(test_size, float):
        if int(np.floor(test_size * n_samples)) < 1:
            ts = 1
        else:
            ts = test_size
    else:
        ts = test_size

    n_splits_adj = min(cv_splits, max(2, n_samples))

    return ShuffleSplit(
        n_splits=n_splits_adj,
        test_size=ts,
        random_state=random_state
    )

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

MAIN_OUTPUT_DIR = "Datewise_Results"

os.makedirs(
    MAIN_OUTPUT_DIR,
    exist_ok=True
)

# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 90)
print(" ADVANCED MACHINE LEARNING — INDEX vs YIELD ")
print("=" * 90)

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'"
    )

# =============================================================================
# LOAD SHEET
# =============================================================================

excel_file = pd.ExcelFile(DATA_FILE)

print("\nAvailable Sheets:\n")

print(excel_file.sheet_names)

sheet_name = "2024-25"

print(f"\n✓ Automatically using sheet: {sheet_name}")

df = pd.read_excel(
    DATA_FILE,
    sheet_name=sheet_name
)

# =============================================================================
# CLEAN COLUMN NAMES
# =============================================================================

df.columns = df.columns.str.strip()

print("\n✓ Available Columns:\n")

print(df.columns.tolist())

print(
    f"\n✓ Loaded dataset: "
    f"{df.shape[0]} rows × {df.shape[1]} columns"
)

# =============================================================================
# FIX DATE COLUMN
# =============================================================================

date_col = "Date"

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

# =============================================================================
# DETECT YEAR
# =============================================================================

available_years = sorted(
    df[date_col].dt.year.dropna().unique()
)

selected_year = available_years[0]

print(f"\n✓ Detected Year: {selected_year}")

# =============================================================================
# AVAILABLE DATES
# =============================================================================

available_dates = sorted(

    df[date_col]
    .dt.strftime("%d-%b")
    .dropna()
    .unique()
)

print("\n✓ Available Dates:\n")

print(available_dates)

# =============================================================================
# USER INPUT DATE
#   Prefer a command-line argument (e.g. `python ... 08-Mar`), else prompt.
#   This also makes automated / non-interactive runs reliable.
# =============================================================================

if len(sys.argv) > 1:
    selected_day = sys.argv[1].strip()
    print(f"\n✓ Using date from argument: {selected_day}")
else:
    selected_day = input(
        "\nEnter date (example: 9-Feb): "
    ).strip()

# =============================================================================
# DATEWISE OUTPUT FOLDER
# =============================================================================

safe_date = (
    selected_day
    .replace("/", "-")
    .replace("\\", "-")
    .replace(" ", "_")
)

DATE_FOLDER = os.path.join(
    MAIN_OUTPUT_DIR,
    safe_date
)

os.makedirs(
    DATE_FOLDER,
    exist_ok=True
)

OUTPUT_EXCEL = os.path.join(
    DATE_FOLDER,
    f"{safe_date}_advanced_ml_results.xlsx"
)

# =============================================================================
# PARSE DATE
# =============================================================================

target_date = pd.to_datetime(
    selected_day,
    format="%d-%b",
    errors="coerce"
)

if pd.isna(target_date):

    target_date = pd.to_datetime(
        selected_day,
        errors="coerce"
    )

if pd.isna(target_date):

    raise ValueError(
        f"Invalid date format: {selected_day}"
    )

target_day = target_date.day
target_month = target_date.month

# =============================================================================
# FILTER DATA
# =============================================================================

day_data = df[

    (df[date_col].dt.year == selected_year)

    &

    (df[date_col].dt.day == target_day)

    &

    (df[date_col].dt.month == target_month)

].copy()

if day_data.empty:

    raise ValueError(
        f"No data found for {selected_day}"
    )

print(
    f"\n✓ Samples found: {len(day_data)}"
)

# =============================================================================
# CREATE logM INDEX
# =============================================================================

eps = 1e-10

required_cols = [
    "NIR",
    "Green",
    "Red edge",
    "Red",
    "Blue"
]

if all(col in day_data.columns for col in required_cols):

    day_data["logM"] = (

        np.log(day_data["NIR"] + eps)

        + np.log(day_data["Green"] + eps)

        + np.log(day_data["Red edge"] + eps)

        - np.log(day_data["Red"] + eps)

        - np.log(day_data["Blue"] + eps)
    )

    print("✓ Created logM index")

# =============================================================================
# CHECK TARGET COLUMN
# =============================================================================

if TARGET_COL not in day_data.columns:

    raise ValueError(
        f"\n'{TARGET_COL}' column not found."
    )

# =============================================================================
# FEATURE SELECTION
# =============================================================================

exclude_keywords = [

    "GY",
    "Yield",
    "BY",
    "kg",
    "plot",
    "rep",
    "block"
]

numeric_cols = day_data.select_dtypes(
    include=[np.number]
).columns.tolist()

feature_names = []

for col in numeric_cols:

    skip = False

    for word in exclude_keywords:

        if word.lower() in col.lower():

            skip = True
            break

    if not skip:

        feature_names.append(col)

print("\n✓ Spectral Indices:\n")

print(feature_names)

# =============================================================================
# REMOVE NaNs
# =============================================================================

valid_cols = feature_names + [TARGET_COL]

day_data = day_data[
    valid_cols
].dropna()

# =============================================================================
# SINGLE INDEX ANALYSIS
# =============================================================================

results = []

print("\nCalculating single index regression...\n")

for feature in feature_names:

    try:

        X = day_data[[feature]]

        y = day_data[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        model = LinearRegression()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, y_pred)
        )

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        results.append({

            "Index": feature,

            "R2_Score":
                round(r2, 4),

            "RMSE":
                round(rmse, 4),

            "MAE":
                round(mae, 4),

            "Coefficient":
                round(model.coef_[0], 4),

            "Intercept":
                round(model.intercept_, 4)
        })

    except:
        continue

if len(results) == 0:
    raise ValueError(
        "No valid single-index regressions were generated."
    )


results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "R2_Score",
    ascending=False
)

# =============================================================================
# COMBINATION ANALYSIS
# =============================================================================

combo_results = []

print("\nFinding best index combinations...\n")

for r in range(

    2,

    MAX_COMBINATION_SIZE + 1
):

    combos_list = list(
        combinations(feature_names, r)
    )

    for combo in combos_list:

        try:

            X_combo = day_data[
                list(combo)
            ]

            y = day_data[TARGET_COL]

            X_train, X_test, y_train, y_test = train_test_split(

                X_combo,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            model = LinearRegression()

            model.fit(X_train, y_train)

            y_pred = model.predict(
                X_test
            )

            r2 = r2_score(
                y_test,
                y_pred
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test,
                    y_pred
                )
            )

            combo_results.append({

                "Combination":
                    " + ".join(combo),

                "N_Indices":
                    len(combo),

                "R2_Score":
                    round(r2, 4),

                "RMSE":
                    round(rmse, 4)
            })

        except:
            continue

if len(combo_results) == 0:
    raise ValueError(
        "No valid feature combinations found."
    )

combo_df = pd.DataFrame(
    combo_results
)

combo_df = combo_df.sort_values(
    "R2_Score",
    ascending=False
)

# =============================================================================
# TOP RESULTS
# =============================================================================

print("\n" + "=" * 90)
print(" TOP INDICES ")
print("=" * 90)

print(
    results_df.head(10)
    .to_string(index=False)
)

print("\n" + "=" * 90)
print(" TOP INDEX COMBINATIONS ")
print("=" * 90)

print(
    combo_df.head(10)
    .to_string(index=False)
)




# =============================================================================
# MACHINE LEARNING MODELS
# =============================================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=10,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=10,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),

    "SVR":
        SVR(
            C=1.0,
            gamma="scale",
            epsilon=0.1
        )
}

if xgb_available:

    models["XGBoost"] = XGBRegressor(
        n_estimators=10,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE
    )

# =============================================================================
# MODEL TRAINING
# =============================================================================

model_results = []

model_combo_results = []

best_overall_r2 = -999

best_train_r2 = -999

best_r2_diff = 999

best_model = None

best_model_name = None

best_predictions = None

best_train_predictions = None

best_features = None

best_y_test = None

best_y_train = None

best_overall_model = None

best_overall_features = None

best_overall_cv = None

best_overall_train_r2 = None

best_overall_test_r2 = None

best_overall_gap = None

cv_strategy = get_cv_strategy(
    len(day_data)
)

print("\nTraining ML models...\n")

for model_name, model in models.items():

    print(f"\nSearching best combination for {model_name}...")

    best_model_r2 = -999

    best_model_combo = None

    best_model_features = None

    best_model_test_pred = None

    best_model_train_pred = None

    best_model_ytest = None

    best_model_ytrain = None

    best_model_train_r2 = None

    best_model_test_r2 = None

    best_model_gap = None

    best_model_cv = np.nan

    best_model_fitted = None

    for combo in combo_df["Combination"]:

        try:
            features = combo.split(" + ")

            X = day_data[features]
            y = day_data[TARGET_COL]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            if model_name == "SVR":
                pipeline = make_pipeline(
                    StandardScaler(),
                    SVR(
                        C=1.0,
                        gamma="scale",
                        epsilon=0.1
                    )
                )

                cv_scores = cross_val_score(
                    pipeline,
                    X,
                    y,
                    cv=cv_strategy,
                    scoring="r2"
                )

                pipeline.fit(
                    X_train,
                    y_train
                )

                y_pred = pipeline.predict(
                    X_test
                )

                train_pred = pipeline.predict(
                    X_train
                )

                fitted = pipeline
            else:
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv_strategy,
                    scoring="r2"
                )

                fitted = clone(model)

                fitted.fit(
                    X_train,
                    y_train
                )

                y_pred = fitted.predict(
                    X_test
                )

                train_pred = fitted.predict(
                    X_train
                )

            avg_cv = np.mean(
                cv_scores
            )

            test_r2 = r2_score(
                y_test,
                y_pred
            )

            train_r2 = r2_score(
                y_train,
                train_pred
            )

            r2_gap = abs(
                train_r2 - test_r2
            )

            if (
                test_r2 > best_model_r2
                and r2_gap <= MAX_TRAIN_TEST_DIFF
            ):
                best_model_r2 = test_r2
                best_model_combo = combo
                best_model_features = features
                best_model_test_pred = y_pred
                best_model_train_pred = train_pred
                best_model_ytest = y_test
                best_model_ytrain = y_train
                best_model_train_r2 = train_r2
                best_model_test_r2 = test_r2
                best_model_gap = r2_gap
                best_model_cv = avg_cv
                best_model_fitted = fitted

        except Exception:
            continue

    # Track whether the model found ANY acceptable combination (gap
    # criterion met). If not, its best score stays at the sentinel -999.
    model_found = best_model_r2 != -999

    model_combo_results.append({
        "Model": model_name,
        "Best_Combination": best_model_combo,
        "N_Features": len(best_model_features)
        if best_model_features is not None
        else 0,
        "Best_R2": round(best_model_r2, 4)
        if model_found
        else np.nan,
        "CV_R2": round(best_model_cv, 4)
        if not pd.isna(best_model_cv)
        else np.nan,
        "Train_R2": round(best_model_train_r2, 4)
        if best_model_train_r2 is not None
        else np.nan,
        "Test_R2": round(best_model_test_r2, 4)
        if best_model_test_r2 is not None
        else np.nan,
        "Gap": round(best_model_gap, 4)
        if best_model_gap is not None
        else np.nan,
        "Best_Features": ", ".join(best_model_features)
        if best_model_features is not None
        else "None"
    })

    # Only consider a model a serious contender if it actually met the
    # gap constraint. This prevents a high-var / overfitting model from
    # being crowned the winner purely on test R2.
    if model_found and best_model_r2 > best_overall_r2:

        best_overall_r2 = best_model_r2

        best_model_name = model_name

        best_overall_model = model_name

        best_predictions = best_model_test_pred

        best_train_predictions = best_model_train_pred

        best_features = best_model_features

        best_overall_features = best_model_features

        best_y_test = best_model_ytest

        best_y_train = best_model_ytrain

        best_overall_cv = best_model_cv

        best_overall_train_r2 = best_model_train_r2

        best_overall_test_r2 = best_model_test_r2

        best_overall_gap = best_model_gap

        best_model = best_model_fitted

    # Report score regardless of gap status
    model_results.append({

        "Model": model_name,

        "R2_Score": round(best_model_r2, 4)

    })

    print(
        f"{model_name:<20}"
        f" Best R2 = {best_model_r2:.4f}"
    )

# If no single model produced a combination that satisfied the train/test
# gap constraint, fall back to whichever model had the highest test R2 so
# the downstream reporting/plots still have a valid winner.
if best_model_name is None:
    fallback_r2 = -999
    for row in model_combo_results:
        if row["Best_R2"] is not None and not pd.isna(row["Best_R2"]):
            if row["Best_R2"] > fallback_r2:
                fallback_r2 = row["Best_R2"]
                _fb_name = row["Model"]

    if fallback_r2 == -999:
        raise RuntimeError(
            "No model was able to fit the data. Check the dataset / columns."
        )

    # Re-run selection using the fallback model's recorded best combo.
    best_model_name = _fb_name
    best_overall_model = _fb_name
    best_overall_r2 = fallback_r2

    _fb_row = next(
        r for r in model_combo_results if r["Model"] == _fb_name
    )
    best_overall_features = (
        _fb_row["Best_Features"].split(", ")
        if _fb_row["Best_Features"] not in (None, "None")
        else None
    )
    best_features = best_overall_features

    # Recompute held-out predictions for the fallback best combo.
    if best_features is not None:
        X = day_data[best_features]
        y = day_data[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        if best_model_name == "SVR":
            fitted = make_pipeline(
                StandardScaler(),
                SVR(C=1.0, gamma="scale", epsilon=0.1)
            )
        else:
            fitted = clone(models[best_model_name])
        fitted.fit(X_train, y_train)
        best_model = fitted
        best_y_test = y_test
        best_y_train = y_train
        best_train_predictions = fitted.predict(X_train)
        best_predictions = fitted.predict(X_test)
        best_overall_test_r2 = r2_score(y_test, best_predictions)
        best_overall_train_r2 = r2_score(y_train, best_train_predictions)
        best_overall_gap = abs(
            best_overall_train_r2 - best_overall_test_r2
        )
        best_overall_cv = np.nan

# -----------------------------------------------------------------------------
# MLFLOW — LOG FINAL METRICS
# -----------------------------------------------------------------------------

if MLFLOW_ENABLED:
    try:
        mlflow.log_params({
            "date": selected_day,
            "year": str(selected_year),
            "best_model": best_overall_model,
            "n_samples": int(len(day_data)),
            "best_features": ", ".join(best_overall_features or [])
        })
        mlflow.log_metrics({
            "cv_r2": float(best_overall_cv) if best_overall_cv is not None else float("nan"),
            "train_r2": float(best_overall_train_r2) if best_overall_train_r2 is not None else float("nan"),
            "test_r2": float(best_overall_test_r2) if best_overall_test_r2 is not None else float("nan"),
            "train_test_gap": float(best_overall_gap) if best_overall_gap is not None else float("nan"),
        })
        mlflow.end_run()
    except Exception as _mle:
        print(f"[WARN] Could not log to MLflow: {_mle}")

model_results_df = pd.DataFrame(
    model_results
)

model_combo_results_df = pd.DataFrame(
    model_combo_results
)

model_results_df = model_results_df.sort_values(
    "R2_Score",
    ascending=False
)

model_combo_results_df = model_combo_results_df.sort_values(
    "Best_R2",
    ascending=False
)
# =============================================================================
# BEST MODEL
# =============================================================================

print("\n" + "=" * 90)

print(f"\n✓ Best ML Model: {best_model_name}")

print(f"✓ Best R2 Score: {best_overall_r2:.4f}")

print(f"✓ Best Features: {best_features}")

print("\n" + "=" * 90)

# =============================================================================
# GRAPH 1 — TOP INDICES
# =============================================================================

top_indices = results_df.head(15)

plt.figure(figsize=(12, 6))

plt.bar(

    top_indices["Index"],

    top_indices["R2_Score"]
)

plt.xticks(rotation=45)

plt.ylabel("R2 Score")

plt.title(
    f"Top Indices Predicting Yield\n{selected_day}-{selected_year}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATE_FOLDER,
        f"{safe_date}_top_indices_r2.png"
    ),
    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 2 — HEATMAP
# =============================================================================

top_features = results_df.head(10)[
    "Index"
].tolist()

heatmap_data = day_data[
    top_features + [TARGET_COL]
]

corr_matrix = heatmap_data.corr()

plt.figure(figsize=(12, 8))

sns.heatmap(

    corr_matrix,

    annot=True,

    cmap="RdYlGn",

    center=0
)

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATE_FOLDER,
        f"{safe_date}_yield_heatmap.png"
    ),
    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 3 — BEST COMBINATIONS
# =============================================================================

top_combo = combo_df.head(10)

plt.figure(figsize=(14, 6))

plt.bar(

    top_combo["Combination"],

    top_combo["R2_Score"]
)

plt.xticks(rotation=60)

plt.ylabel("R2 Score")

plt.title(
    "Best Index Combinations"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATE_FOLDER,
        f"{safe_date}_best_combinations.png"
    ),
    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 4 — TEST ACTUAL vs PREDICTED
# =============================================================================

plt.figure(figsize=(10,6))

plt.scatter(
    best_y_test.values,
    best_predictions
)

min_val = min(
    min(best_y_test.values),
    min(best_predictions)
)

max_val = max(
    max(best_y_test.values),
    max(best_predictions)
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Yield (kg/ha)")

plt.ylabel("Predicted Yield (kg/ha)")

plt.title(
    f"{best_model_name}\nR²={best_overall_r2:.4f}"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATE_FOLDER,
        f"{safe_date}_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 4B — TRAINING ACTUAL vs PREDICTED
# =============================================================================

# Training R2
train_r2 = r2_score(
    best_y_train,
    best_train_predictions
)

plt.figure(figsize=(10,6))

plt.scatter(
    best_y_train.values,
    best_train_predictions
)

min_val = min(
    min(best_y_train.values),
    min(best_train_predictions)
)

max_val = max(
    max(best_y_train.values),
    max(best_train_predictions)
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual Training Yield (kg/ha)")

plt.ylabel("Predicted Training Yield (kg/ha)")

plt.title(
    f"Training Data — {best_model_name}\nR² = {train_r2:.4f}"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        DATE_FOLDER,
        f"{safe_date}_training_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 4C — TRAINING RESIDUALS
# =============================================================================

residuals = (

    best_y_train.values

    - best_train_predictions
)

plt.figure(figsize=(10,6))

plt.scatter(

    best_train_predictions,

    residuals
)

plt.axhline(

    y=0,

    linestyle="--"
)

plt.xlabel(

    "Predicted Yield (kg/ha)"
)

plt.ylabel(

    "Residuals"
)

plt.title(

    f"Training Residual Plot - {best_model_name}"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        DATE_FOLDER,

        f"{safe_date}_training_residuals.png"
    ),

    dpi=300
)

plt.close()

# =============================================================================
# GRAPH 5 — FEATURE IMPORTANCE
# =============================================================================

# Feature importances are read ONLY from the fitted model that was actually
# used for the winning combination (best_model). No refitting on the full
# dataset -> no data leakage. Linear Regression / SVR have no
# feature_importances_, so the plot is simply skipped (no crash).
importance = None

if (
    best_model is not None
    and hasattr(best_model, "feature_importances_")
    and best_features is not None
    and len(best_features) > 0
):
    try:
        importance = best_model.feature_importances_

        if len(importance) != len(best_features):
            importance = None
    except Exception:
        importance = None

if importance is not None:
    importance_df = pd.DataFrame({
        "Feature": best_features,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.bar(

        importance_df["Feature"],

        importance_df["Importance"]
    )

    plt.xticks(rotation=45)

    plt.ylabel("Importance")

    plt.title(
        f"Feature Importance ({best_model_name})"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            DATE_FOLDER,
            f"{safe_date}_feature_importance.png"
        ),
        dpi=300
    )

    plt.close()

# =============================================================================
# SAVE TEST PREDICTIONS
# =============================================================================

prediction_df = pd.DataFrame({

    "Actual_Yield":
        best_y_test.values,

    "Predicted_Yield":
        best_predictions
})
# =============================================================================
# SAVE TEST RESIDUALS
# =============================================================================

test_residuals_df = pd.DataFrame({

    "Actual_Yield":
        best_y_test.values,

    "Predicted_Yield":
        best_predictions,

    "Residual":

        best_y_test.values
        -
        best_predictions
})
# =============================================================================
# SAVE TRAINING PREDICTIONS
# =============================================================================

train_prediction_df = pd.DataFrame({

    "Actual_Yield":
        best_y_train.values,

    "Predicted_Yield":
        best_train_predictions
})


train_residuals_df = pd.DataFrame({

    "Actual_Yield":
        best_y_train.values,

    "Predicted_Yield":
        best_train_predictions,

    "Residual":

        best_y_train.values
        -
        best_train_predictions
})

train_metrics_df = pd.DataFrame({

    "Metric": [

        "Training_R2",

        "Training_RMSE",

        "Training_MAE"
    ],

    "Value": [

        r2_score(
            best_y_train,
            best_train_predictions
        ),

        np.sqrt(
            mean_squared_error(
                best_y_train,
                best_train_predictions
            )
        ),

        mean_absolute_error(
            best_y_train,
            best_train_predictions
        )
    ]
})
# =============================================================================
# TESTING METRICS
# =============================================================================

test_metrics_df = pd.DataFrame({

    "Metric": [

        "Testing_R2",

        "Testing_RMSE",

        "Testing_MAE"
    ],

    "Value": [

        r2_score(
            best_y_test,
            best_predictions
        ),

        np.sqrt(
            mean_squared_error(
                best_y_test,
                best_predictions
            )
        ),

        mean_absolute_error(
            best_y_test,
            best_predictions
        )
    ]
})
# =============================================================================
# SAVE EXCEL
# =============================================================================

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    results_df.to_excel(

        writer,

        sheet_name="Single_Indices",

        index=False
    )

    model_combo_results_df.to_excel(

    writer,

    sheet_name="Best_Model_Combinations",

    index=False
)

    combo_df.to_excel(

        writer,

        sheet_name="Best_Combinations",

        index=False
    )

    model_results_df.to_excel(

        writer,

        sheet_name="ML_Model_Results",

        index=False
    )

    prediction_df.to_excel(

        writer,

        sheet_name="Predictions",

        index=False
    )

    train_prediction_df.to_excel(

        writer,

        sheet_name="Training_Predictions",

        index=False
    )

    train_metrics_df.to_excel(

    writer,

    sheet_name="Training_Metrics",

    index=False
)
    test_metrics_df.to_excel(

    writer,

    sheet_name="Testing_Metrics",

    index=False
)

train_residuals_df.to_excel(

    writer,

    sheet_name="Training_Residuals",

    index=False
)

test_residuals_df.to_excel(

    writer,

    sheet_name="Testing_Residuals",

    index=False
)

# =============================================================================
# RAG AGENT INTEGRATION
# =============================================================================
# Build a context dict from the computed ML results and feed it to the RAG
# agent. The agent builds a retrievable knowledge base, generates a natural
# language report, and can run an interactive chat about the analysis.
# =============================================================================

def build_rag_context():
    return {
        "date": selected_day,
        "year": selected_year,
        "n_samples": len(day_data),
        "best_model": best_overall_model,
        "best_cv_r2": None if best_overall_cv is None else float(best_overall_cv),
        "best_train_r2": None if best_overall_train_r2 is None else float(best_overall_train_r2),
        "best_test_r2": None if best_overall_test_r2 is None else float(best_overall_test_r2),
        "best_gap": None if best_overall_gap is None else float(best_overall_gap),
        "best_features": best_overall_features,
        "models": model_combo_results_df.to_dict("records")
        if isinstance(model_combo_results_df, pd.DataFrame) else [],
        "top_indices": results_df.head(15).to_dict("records")
        if isinstance(results_df, pd.DataFrame) else [],
        "combinations": combo_df.head(10).to_dict("records")
        if isinstance(combo_df, pd.DataFrame) else [],
    }

if RAG_USE_AGENT:
    try:
        from rag_agent import RAGAgent, run_interactive_chat, setup_hint

        rag_context = build_rag_context()
        rag = RAGAgent(provider=RAG_PROVIDER)
        n_chunks = rag.ingest(rag_context)

        print("\n" + "=" * 90)
        print(f" RAG AGENT ACTIVE (provider: {RAG_PROVIDER.upper()}, {n_chunks} knowledge chunks)")
        print("=" * 90)

        # Agent recommendation on how to make the ML perform well
        print("\nRAG Agent - ML Performance Recommendation:")
        for rec in rag.agent_decision():
            print("   • " + rec)

        # Generate a full report and save it next to the other outputs
        try:
            report = rag.generate_report()
            REPORT_PATH = os.path.join(DATE_FOLDER, f"{safe_date}_rag_report.txt")
            with open(REPORT_PATH, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\n✓ RAG report saved: {REPORT_PATH}")
        except Exception as e:
            print(f"\n[WARN] Could not generate RAG report: {e}")

        # Optional interactive chat
        if RAG_INTERACTIVE:
            run_interactive_chat(rag)

    except Exception as e:
        print(f"\n[WARN] RAG agent unavailable, continuing without it: {e}")

# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 90)

print(
    f"\n✓ All files saved inside:\n{DATE_FOLDER}"
)

print(f"✓ Saved: {safe_date}_top_indices_r2.png")

print(f"✓ Saved: {safe_date}_yield_heatmap.png")

print(f"✓ Saved: {safe_date}_best_combinations.png")

print(f"✓ Saved: {safe_date}_actual_vs_predicted.png")

print(f"✓ Saved: {safe_date}_training_actual_vs_predicted.png")

print(f"✓ Saved: {safe_date}_training_residuals.png")

if hasattr(best_model, "feature_importances_"):

    print(
        f"✓ Saved: {safe_date}_feature_importance.png"
    )

print(
    f"\n✓ Files saved inside:\n{DATE_FOLDER}"
)

print("\n" + "=" * 90)

print("\n✓ MACHINE LEARNING ANALYSIS COMPLETED")

print("\n" + "=" * 90)

print("\n" + "=" * 90)

print(
    f"\nBEST MODEL : {best_overall_model}"
)

print(
    f"CV R\u00b2 : {best_overall_cv:.4f}"
    if best_overall_cv is not None else "CV R\u00b2 : N/A"
)

print(
    f"Train R\u00b2 : {best_overall_train_r2:.4f}"
    if best_overall_train_r2 is not None else "Train R\u00b2 : N/A"
)

print(
    f"Test R\u00b2 : {best_overall_test_r2:.4f}"
    if best_overall_test_r2 is not None else "Test R\u00b2 : N/A"
)

print(
    f"Gap : {best_overall_gap:.4f}"
    if best_overall_gap is not None else "Gap : N/A"
)

print(
    f"Features : {best_overall_features}"
)

print("\n" + "=" * 90)
# =============================================================================
# VERIFY FILES
# =============================================================================

print("\nCurrent Working Directory:")
print(os.getcwd())

print("\nContents of Datewise_Results:\n")

for root, dirs, files in os.walk("Datewise_Results"):
    print(root)
    for f in files:
        print("   ", f)

# =============================================================================
# CREATE ZIP FILE
# =============================================================================

ZIP_NAME = f"{safe_date}_Results"

ZIP_PATH = shutil.make_archive(
    ZIP_NAME,
    "zip",
    DATE_FOLDER
)

print(f"\n✓ ZIP File Created: {ZIP_NAME}.zip")

# =============================================================================
# FINAL LOCAL ZIP OUTPUT
# =============================================================================

zip_full_path = os.path.abspath(f"{ZIP_NAME}.zip")

print("\n" + "=" * 90)
print("✓ Running on Local Machine (VS Code)")
print(f"✓ ZIP File Created: {zip_full_path}")
print("\n✓ All results are zipped and available at:")
print(f"   {zip_full_path}")
print("\n" + "=" * 90)
print(" ANALYSIS COMPLETE - ALL OUTPUTS GENERATED ")
print("=" * 90)
print("\n✓ Generated Files:")
print("   • Single Indices Analysis (Excel)")
print("   • Best Model Combinations (Excel)")
print("   • ML Model Results (Excel)")
print("   • Training & Testing Predictions (Excel)")
print("   • Training & Testing Residuals (Excel)")
print("   • Training & Testing Metrics (Excel)")
print("   • Top Indices Chart (PNG)")
print("   • Feature Correlation Heatmap (PNG)")
print("   • Best Combinations Chart (PNG)")
print("   • Actual vs Predicted Plots (PNG)")
print("   • Training Residuals Plot (PNG)")
if hasattr(best_model, "feature_importances_"):
    print(f"   • Feature Importance Chart (PNG)")
print(f"\n✓ Results Folder: {DATE_FOLDER}")
print(f"✓ ZIP Archive: {ZIP_NAME}.zip")
print("\n" + "=" * 90)
