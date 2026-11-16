#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 generate_charts.py
python3 ui/app.py
