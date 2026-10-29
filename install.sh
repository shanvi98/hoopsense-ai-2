#!/bin/bash
# HoopSense AI — Smart Mac Installer
# Installs in layers so a CV failure doesn't block the whole app

set -e
cd "$(dirname "$0")/backend"

echo "🏀 HoopSense AI — Installing dependencies"
echo ""

# Activate venv if not already active
if [[ -z "$VIRTUAL_ENV" ]]; then
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
        echo "✅ Created .venv"
    fi
    source .venv/bin/activate
    echo "✅ Activated .venv"
fi

echo ""
echo "── Step 1: Core dependencies (FastAPI, ML, NumPy) ──"
pip install -q --upgrade pip
pip install -r requirements-core.txt
echo "✅ Core dependencies installed"

echo ""
echo "── Step 2: Computer vision libraries ──"
echo "   (These may take a while — skipping any that fail)"

pip install "opencv-python-headless>=4.8.0" && echo "✅ OpenCV" || echo "⚠️  OpenCV failed — app will run in demo mode"

pip install "mediapipe>=0.10.0" && echo "✅ MediaPipe" || echo "⚠️  MediaPipe failed — app will use synthetic pose data"

pip install "ultralytics>=8.0.0" && echo "✅ YOLOv8" || echo "⚠️  YOLOv8 failed — app will use synthetic ball detection"

echo ""
echo "── Step 3: Training models ──"
mkdir -p data/processed data/models

python3 scripts/train_models.py \
    --synthetic \
    --data data/processed/shots.csv \
    --output data/models/ && echo "✅ Models trained" || echo "⚠️  Training failed — app uses rule-based scoring"

echo ""
echo "══════════════════════════════════════════"
echo "✅ Installation complete!"
echo ""
echo "Start backend:   source .venv/bin/activate && uvicorn app.main:app --reload --port 8000"
echo "Start frontend:  cd ../frontend && npm install && npm run dev"
echo ""
echo "Open: http://localhost:5173"
echo "══════════════════════════════════════════"
