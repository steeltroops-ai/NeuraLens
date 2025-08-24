#!/bin/bash
# NeuraLens Backend Environment Setup Script for macOS/Linux
# This script sets up the Python virtual environment and installs dependencies

set -e  # Exit on any error

echo "========================================"
echo "NeuraLens Backend Environment Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.9+ from your package manager or https://python.org"
    exit 1
fi

echo "Python version:"
python3 --version
echo

# Check if we're in the backend directory
if [ ! -f "requirements.txt" ]; then
    echo "ERROR: requirements.txt not found"
    echo "Please run this script from the backend directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created successfully!"
    echo
else
    echo "Virtual environment already exists."
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip
echo

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo
echo "To deactivate the environment, run:"
echo "  deactivate"
echo
echo "To start the development server, run:"
echo "  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo

# Make the script executable
chmod +x "$0"
