#!/usr/bin/env bash
# =========================================================================
#  ML Yield Prediction - one-click launcher (macOS / Linux)
#  Usage:
#    ./run.sh                 -> prompts for date (interactive)
#    ./run.sh 08-Mar          -> full run on a date
#    ./run.sh 07-Feb --smoke  -> fast smoke run on a date
# =========================================================================
set -e
cd "$(dirname "$0")"

SCRIPT="Latest Updated Code for IDLE.py"
DATE="${1:-}"

echo
echo " ============================================================"
echo "  ML Yield Prediction"
echo " ============================================================"
echo

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Could not find $SCRIPT in $(pwd)"
    exit 1
fi

if [ ! -f "Bands&VI data_ML.xlsx" ]; then
    echo "[ERROR] Data file 'Bands&VI data_ML.xlsx' not found in: $(pwd)"
    echo "        Keep it in the same folder as the script."
    exit 1
fi

# Shift off the date so $@ holds the remaining args (e.g. --smoke)
shift || true
python "$SCRIPT" $DATE "$@"
