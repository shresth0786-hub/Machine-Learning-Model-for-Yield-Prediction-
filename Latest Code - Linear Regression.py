# =============================================================================
# ADVANCED MACHINE LEARNING: INDEX vs YIELD ANALYSIS (LOCAL VS CODE VERSION)
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns

# Force matplotlib to use a non-interactive backend safe for local scripts
import matplotlib
matplotlib.use('Agg')
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# OPTIONAL XGBOOST
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "Bands&VI data_ML.xlsx"
TARGET_COL = "GY (kg/ha)"
MAX_COMBINATION_SIZE = 3
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 20
MAX_TRAIN_TEST_DIFF = 0.30

def get_cv_strategy(n_samples, cv_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    if isinstance(test_size, float):
        ts = 1 if int(np.floor(test_size * n_samples)) < 1 else test_size
    else:
        ts = test_size
    n_splits_adj = min(cv_splits, max(2, n_samples))
    return ShuffleSplit(n_splits=n_splits_adj, test_size=ts, random_state=random_state)

# =============================================================================
# OUTPUT DIRECTORY SETUP (LOCAL WORKSPACE)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
MAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "Datewise_Results")
os.makedirs(MAIN_OUTPUT_DIR, exist_ok=True)

print("\n" + "=" * 90)
print(" ADVANCED MACHINE LEARNING — INDEX vs YIELD (VS CODE)")
print("=" * 90)

LOCAL_DATA_PATH = os.path.join(BASE_DIR, DATA_FILE)
if not os.path.exists(LOCAL_DATA_PATH):
    raise FileNotFoundError(f"Could not find '{DATA_FILE}' at {LOCAL_DATA_PATH}. Please place it in the same directory.")

# =============================================================================
# LOAD SHEET & DATES
# =============================================================================

excel_file = pd.ExcelFile(LOCAL_DATA_PATH)
sheet_name = "2024-25"
df = pd.read_excel(LOCAL_DATA_PATH, sheet_name=sheet_name)
df.columns = df.columns.str.strip()

date_col = "Date"
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
selected_year = sorted(df[date_col].dt.year.dropna().unique())[0]

available_dates = sorted(df[date_col].dt.strftime("%d-%b").dropna().unique())
print(f"\n✓ Available Dates: {available_dates}")

# USER INPUT WITH DEFAULT VALUE FOR EASIER VS CODE TESTING
user_input = input("\nEnter date (example: 08-Mar) [Press Enter for default '08-Mar']: ").strip()
selected_day = user_input if user_input else "08-Mar"

safe_date = selected_day.replace("/", "-").replace("\\", "-").replace(" ", "_")
DATE_FOLDER = os.path.join(MAIN_OUTPUT_DIR, safe_date)
os.makedirs(DATE_FOLDER, exist_ok=True)
OUTPUT_EXCEL = os.path.join(DATE_FOLDER, f"{safe_date}_advanced_ml_results.xlsx")

target_date = pd.to_datetime(selected_day, format="%d-%b", errors="coerce") or pd.to_datetime(selected_day, errors="coerce")
if pd.isna(target_date):
    raise ValueError(f"Invalid date format: {selected_day}")

day_data = df[
    (df[date_col].dt.year == selected_year) &
    (df[date_col].dt.day == target_date.day) &
    (df[date_col].dt.month == target_date.month)
].copy()

if day_data.empty:
    raise ValueError(f"No data found for {selected_day}")

print(f"✓ Samples found: {len(day_data)}")

# =============================================================================
# DATA PROCESSING & LOGM INDEX
# =============================================================================

eps = 1e-10
required_cols = ["NIR", "Green", "Red edge", "Red", "Blue"]
if all(col in day_data.columns for col in required_cols):
    day_data["logM"] = (
        np.log(day_data["NIR"] + eps) + np.log(day_data["Green"] + eps) +
        np.log(day_data["Red edge"] + eps) - np.log(day_data["Red"] + eps) -
        np.log(day_data["Blue"] + eps)
    )

exclude_keywords = ["GY", "Yield", "BY", "kg", "plot", "rep", "block"]
numeric_cols = day_data.select_dtypes(include=[np.number]).columns.tolist()
feature_names = [col for col in numeric_cols if not any(word in col.lower() for word in exclude_keywords)]

day_data = day_data[feature_names + [TARGET_COL]].dropna()

# =============================================================================
# INDICES ANALYSIS
# =============================================================================

results = []
for feature in feature_names:
    try:
        X, y = day_data[[feature]], day_data[TARGET_COL]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({
            "Index": feature, "R2_Score": round(r2_score(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "Coefficient": round(model.coef_[0], 4), "Intercept": round(model.intercept_, 4)
        })
    except: continue
results_df = pd.DataFrame(results).sort_values("R2_Score", ascending=False)

combo_results = []
for r in range(2, MAX_COMBINATION_SIZE + 1):
    for combo in combinations(feature_names, r):
        try:
            X_combo, y = day_data[list(combo)], day_data[TARGET_COL]
            X_train, X_test, y_train, y_test = train_test_split(X_combo, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            combo_results.append({
                "Combination": " + ".join(combo), "N_Indices": len(combo),
                "R2_Score": round(r2_score(y_test, y_pred), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            })
        except: continue
combo_df = pd.DataFrame(combo_results).sort_values("R2_Score", ascending=False)

# =============================================================================
# MACHINE LEARNING ENGINE (STABLE CV CONFIGURATION)
# =============================================================================

models = {"Linear Regression": LinearRegression()}
model_results, model_combo_results = [], []

best_overall_cv_r2 = -999
best_model, best_model_name, best_features = None, None, None
best_predictions, best_train_predictions = None, None
best_y_test, best_y_train = None, None
best_overall_cv, best_overall_train_r2, best_overall_test_r2, best_overall_gap = None, None, None, None

cv_strategy = get_cv_strategy(len(day_data))

for model_name, model in models.items():
    best_model_cv = -999
    best_model_combo, best_model_features = None, None
    best_model_test_pred, best_model_train_pred = None, None
    best_model_ytest, best_model_ytrain = None, None
    best_model_train_r2, best_model_test_r2, best_model_gap = None, None, None

    for combo in combo_df["Combination"]:
        try:
            features = combo.split(" + ")
            X, y = day_data[features], day_data[TARGET_COL]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

            current_model = make_pipeline(StandardScaler(), SVR()) if model_name == "SVR" else model
            cv_scores = cross_val_score(current_model, X_train, y_train, cv=cv_strategy, scoring="r2")
            avg_cv = np.mean(cv_scores)

            current_model.fit(X_train, y_train)
            train_pred = current_model.predict(X_train)
            y_pred = current_model.predict(X_test)

            train_r2, test_r2 = r2_score(y_train, train_pred), r2_score(y_test, y_pred)
            r2_gap = abs(train_r2 - test_r2)

            if avg_cv > best_model_cv and r2_gap <= MAX_TRAIN_TEST_DIFF:
                best_model_cv = avg_cv
                best_model_combo = combo
                best_model_features = features
                best_model_test_pred = y_pred
                best_model_train_pred = train_pred
                best_model_ytest = y_test
                best_model_ytrain = y_train
                best_model_train_r2 = train_r2
                best_model_test_r2 = test_r2
                best_model_gap = r2_gap
        except: continue

    model_combo_results.append({
        "Model": model_name, "Best_Combination": best_model_combo,
        "N_Features": len(best_model_features) if best_model_features is not None else 0,
        "Best_R2": round(best_model_test_r2, 4) if best_model_test_r2 is not None else np.nan,
        "CV_R2": round(best_model_cv, 4) if best_model_cv != -999 else np.nan,
        "Train_R2": round(best_model_train_r2, 4) if best_model_train_r2 is not None else np.nan,
        "Test_R2": round(best_model_test_r2, 4) if best_model_test_r2 is not None else np.nan,
        "Gap": round(best_model_gap, 4) if best_model_gap is not None else np.nan,
        "Best_Features": ", ".join(best_model_features) if best_model_features is not None else "None"
    })
    model_results.append({"Model": model_name, "R2_Score": round(best_model_test_r2, 4) if best_model_test_r2 is not None else np.nan})

    if best_model_cv > best_overall_cv_r2:
        best_overall_cv_r2 = best_model_cv
        best_model_name = model_name
        best_predictions = best_model_test_pred
        best_train_predictions = best_model_train_pred
        best_features = best_model_features
        best_y_test = best_model_ytest
        best_y_train = best_model_ytrain
        best_overall_cv = best_model_cv
        best_overall_train_r2 = best_model_train_r2
        best_overall_test_r2 = best_model_test_r2
        best_overall_gap = best_model_gap
        best_model = model

model_results_df = pd.DataFrame(model_results).sort_values("R2_Score", ascending=False)
model_combo_results_df = pd.DataFrame(model_combo_results).sort_values("Best_R2", ascending=False)

# =============================================================================
# EXPORTS & GRAPH GENERATION
# =============================================================================

print("\n" + "=" * 90)
print(f"✓ Best ML Model: {best_model_name}")
print(f"✓ Best CV R2 Score: {best_overall_cv:.4f}")
print(f"✓ Best Test R2 Score: {best_overall_test_r2:.4f}")
print(f"✓ Best Features: {best_features}")
print("=" * 90)

# Save Plot 1
plt.figure(figsize=(12, 6))
plt.bar(results_df.head(15)["Index"], results_df.head(15)["R2_Score"])
plt.xticks(rotation=45); plt.ylabel("R2 Score"); plt.title(f"Top Indices ({selected_day})"); plt.tight_layout()
plt.savefig(os.path.join(DATE_FOLDER, f"{safe_date}_top_indices_r2.png"), dpi=300); plt.close()

# Save Plot 2
plt.figure(figsize=(12, 8))
sns.heatmap(day_data[results_df.head(10)["Index"].tolist() + [TARGET_COL]].corr(), annot=True, cmap="RdYlGn", center=0)
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig(os.path.join(DATE_FOLDER, f"{safe_date}_yield_heatmap.png"), dpi=300); plt.close()

# Save Plot 3
plt.figure(figsize=(14, 6))
plt.bar(combo_df.head(10)["Combination"], combo_df.head(10)["R2_Score"])
plt.xticks(rotation=60); plt.ylabel("R2 Score"); plt.title("Best Index Combinations"); plt.tight_layout()
plt.savefig(os.path.join(DATE_FOLDER, f"{safe_date}_best_combinations.png"), dpi=300); plt.close()

# Save Plot 4 (Actual vs Predicted)
plt.figure(figsize=(10,6))
plt.scatter(best_y_test.values, best_predictions)
m_val = min(min(best_y_test.values), min(best_predictions))
mx_val = max(max(best_y_test.values), max(best_predictions))
plt.plot([m_val, mx_val], [m_val, mx_val], linestyle="--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"{best_model_name} (Test R²={best_overall_test_r2:.4f})"); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(DATE_FOLDER, f"{safe_date}_actual_vs_predicted.png"), dpi=300); plt.close()

# Save Excel Files
with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    results_df.to_excel(writer, sheet_name="Single_Indices", index=False)
    model_combo_results_df.to_excel(writer, sheet_name="Best_Model_Combinations", index=False)
    combo_df.to_excel(writer, sheet_name="Best_Combinations", index=False)
    model_results_df.to_excel(writer, sheet_name="ML_Model_Results", index=False)
    pd.DataFrame({"Actual": best_y_test.values, "Predicted": best_predictions}).to_excel(writer, sheet_name="Predictions", index=False)

# Archive ZIP locally
import shutil
ZIP_NAME = os.path.join(BASE_DIR, f"{safe_date}_Results")
shutil.make_archive(ZIP_NAME, "zip", DATE_FOLDER)

print(f"\n✓ RUN COMPLETED SUCCESSFULLY.")
print(f"✓ All assets and zip generated inside workspace directory: {DATE_FOLDER}")
print("=" * 90)