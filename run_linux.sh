#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 is required. Install Python 3 first."
    exit 1
fi

if [ ! -f best.pt ]; then
    echo "ERROR: best.pt is missing from this project folder."
    exit 1
fi

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export MODEL_PATH="$PWD/best.pt"
python app.py
