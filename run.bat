@echo off
REM =========================================================================
REM  ML Yield Prediction - one-click launcher
REM  Usage:
REM    run.bat                 -> prompts for date (interactive)
REM    run.bat 08-Mar          -> full run on a date
REM    run.bat 07-Feb --smoke  -> fast smoke run on a date
REM =========================================================================
setlocal
cd /d "%~dp0"

set "SCRIPT=Latest Updated Code for IDLE.py"

REM Pick the date argument (or prompt if none given)
if "%~1"=="" (
    set "DATE="
) else (
    set "DATE=%~1"
)

echo.
echo  ============================================================
echo   ML Yield Prediction
echo  ============================================================
echo.

if not exist "%SCRIPT%" (
    echo [ERROR] Could not find %SCRIPT% in %cd%
    pause
    exit /b 1
)

if not exist "Bands&VI data_ML.xlsx" (
    echo [ERROR] Data file 'Bands&VI data_ML.xlsx' not found in: %cd%
    echo         Keep it in the same folder as the script.
    pause
    exit /b 1
)

if "%DATE%"=="" (
    python "%SCRIPT%" %2 %3
) else (
    python "%SCRIPT%" "%DATE%" %2 %3
)

echo.
echo ============================================================
echo  Run finished. Press any key to close.
echo ============================================================
pause >nul
endlocal
