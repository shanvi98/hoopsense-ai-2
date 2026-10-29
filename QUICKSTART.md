# HoopSense AI — Quick Start (Mac)

## Option A: Automated (run one script)

```bash
cd hoopsense-ai
bash setup.sh
```

---

## Option B: Manual step by step

Open **two terminal tabs**.

### Tab 1 — Backend

```bash
cd hoopsense-ai/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/train_models.py --synthetic --data data/processed/shots.csv --output data/models/
uvicorn app.main:app --reload --port 8000
```

### Tab 2 — Frontend

```bash
cd hoopsense-ai/frontend
npm install
npm run dev
```

Then open → http://localhost:5173

---

## Troubleshooting

**`python` not found** → use `python3` instead  
**`pip` not found** → use `pip3` or `python3 -m pip`  
**Port 8000 in use** → `lsof -i :8000` then `kill <PID>`  
**Port 5173 in use** → `lsof -i :5173` then `kill <PID>`  
**OpenCV error on Mac M1/M2** → `pip install opencv-python-headless` (already in requirements)  
**MediaPipe not available** → app runs in demo/synthetic mode automatically, no action needed  
