#!/bin/bash
# Battery RUL Notebook Launcher
# This script starts Jupyter Notebook in the RUL directory

echo "============================================================"
echo "Battery RUL Prediction - Jupyter Notebook Launcher"
echo "============================================================"
echo ""

echo "Starting Jupyter Notebook..."
echo ""
echo "The notebook will open in your default browser."
echo "Press Ctrl+C in this terminal to stop the server when done."
echo ""
echo "============================================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Start Jupyter Notebook
jupyter notebook battery_remaining_life_prediction.ipynb
