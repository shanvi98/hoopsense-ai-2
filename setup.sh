#!/bin/bash
# HoopSense AI — Mac Setup Script
set -e

echo "🏀 HoopSense AI Setup"
echo "====================="

# ── Check Python ──────────────────────────────────────────────────────────────
# MediaPipe ships wheels for Python 3.9–3.12 only; 3.11 matches the Docker image
PYTHON=""
for candidate in python3.11 python3.12 python3.10; do
    if command -v $candidate &>/dev/null; then PYTHON=$candidate; break; fi
done
if [[ -z "$PYTHON" ]]; then
    echo "❌ Python 3.10–3.12 not found (MediaPipe does not support 3.13+)."
    echo "   Install with: brew install python@3.11"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# ── Check Node ────────────────────────────────────────────────────────────────
if command -v node &>/dev/null; then
    echo "✅ Node: $(node --version)"
else
    echo "❌ Node.js not found. Install from https://nodejs.org/"
    exit 1
fi

# ── Backend ───────────────────────────────────────────────────────────────────
echo ""
echo "── Setting up backend ──"
cd backend

$PYTHON -m venv .venv
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt

echo "Training models with synthetic data..."
$PYTHON scripts/train_models.py --synthetic --data data/processed/shots.csv --output data/models/

echo ""
echo "── Starting backend (background) ──"
uvicorn app.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"
cd ..

# ── Frontend ──────────────────────────────────────────────────────────────────
echo ""
echo "── Setting up frontend ──"
cd frontend
npm install
echo ""
echo "── Starting frontend ──"
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ HoopSense AI is running!"
echo "   Frontend → http://localhost:5173"
echo "   Backend  → http://localhost:8000"
echo "   API Docs → http://localhost:8000/api/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
wait $BACKEND_PID $FRONTEND_PID
