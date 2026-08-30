# =============================================================================
# ADVANCED MACHINE LEARNING:
# INDEX vs YIELD ANALYSIS
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

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

MAX_TRAIN_TEST_DIFF = 0.30

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
# =============================================================================

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

models ={

    "SVR":
        SVR(
            C=1.0,
            gamma="scale",
            epsilon=0.1
        )
}

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
                    SVR()
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
            else:
                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv_strategy,
                    scoring="r2"
                )

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
                    X_test
                )

                train_pred = model.predict(
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

        except Exception:
            continue

        model_combo_results.append({
            "Model": model_name,
            "Best_Combination": best_model_combo,
            "N_Features": len(best_model_features)
            if best_model_features is not None
            else 0,
            "Best_R2": round(best_model_r2, 4)
            if best_model_r2 != -999
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



model_results.append({

        "Model": model_name,

        "R2_Score": round(best_model_r2, 4)

    })

print(
        f"{model_name:<20}"
        f" Best R2 = {best_model_r2:.4f}"
    )

if best_model_r2 > best_overall_r2:

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

        best_model = model
# =============================================================================
# RESULTS DATAFRAME
# =============================================================================

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

if hasattr(best_model, "feature_importances_"):

    importance = best_model.feature_importances_

    importance_df = pd.DataFrame({

        "Feature":
            best_features,

        "Importance":
            importance
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
    f"CV R² : {best_overall_cv:.4f}"
)

print(
    f"Train R² : {best_overall_train_r2:.4f}"
)

print(
    f"Test R² : {best_overall_test_r2:.4f}"
)

print(
    f"Gap : {best_overall_gap:.4f}"
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

import shutil

ZIP_NAME = f"{safe_date}_Results"

shutil.make_archive(
    ZIP_NAME,
    "zip",
    DATE_FOLDER
)

print(f"\n✓ ZIP File Created: {ZIP_NAME}.zip")

# =============================================================================
# DOWNLOAD ZIP
# =============================================================================

try:

    from google.colab import files

    files.download(
        f"{ZIP_NAME}.zip"
    )

except Exception:

    print(
        f"ZIP saved as {ZIP_NAME}.zip"
    )
