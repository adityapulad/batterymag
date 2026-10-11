@echo off
REM Battery RUL Notebook Launcher
REM This script starts Jupyter Notebook in the RUL directory

echo ============================================================
echo Battery RUL Prediction - Jupyter Notebook Launcher
echo ============================================================
echo.

echo Starting Jupyter Notebook...
echo.
echo The notebook will open in your default browser.
echo Press Ctrl+C in this window to stop the server when done.
echo.
echo ============================================================
echo.

cd /d "%~dp0"
jupyter notebook battery_remaining_life_prediction.ipynb

pause
