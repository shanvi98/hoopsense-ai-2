# 🏀 HoopSense AI

**Production-grade basketball shooting analysis platform** powered by computer vision and machine learning.

Upload a shooting video → get instant AI-powered form scoring, biomechanical feedback, and shot success probability.

---

## Architecture

```
hoopsense-ai/
├── backend/                        # FastAPI + Python
│   ├── app/
│   │   ├── main.py                 # FastAPI app entry point
│   │   ├── api/
│   │   │   ├── analysis.py         # Upload, poll, demo endpoints
│   │   │   ├── health.py           # Health / readiness probes
│   │   │   └── models.py           # Model registry API
│   │   ├── core/
│   │   │   ├── config.py           # Pydantic settings (env-based)
│   │   │   └── logger.py           # Structured logging
│   │   ├── models/
│   │   │   └── schemas.py          # Pydantic v2 data models
│   │   ├── services/
│   │   │   ├── video_processor.py  # OpenCV frame extraction
│   │   │   ├── pose_estimator.py   # MediaPipe pose estimation
│   │   │   ├── object_detector.py  # YOLOv8 ball/hoop tracking
│   │   │   ├── feature_engineer.py # 20 biomechanical features
│   │   │   ├── inference_engine.py # XGBoost scoring + feedback
│   │   │   └── analysis_service.py # Pipeline orchestration
│   │   └── ml/
│   │       └── model_registry.py   # Model versioning & metadata
│   ├── scripts/
│   │   ├── train_models.py         # XGBoost training + evaluation
│   │   ├── preprocess_dataset.py   # Raw video → feature CSV
│   │   └── evaluate_models.py      # Full eval report + plots
│   ├── data/
│   │   ├── uploads/                # Temp uploaded videos
│   │   ├── processed/              # Feature CSVs
│   │   ├── models/                 # Trained model artifacts
│   │   └── outputs/                # Processed video outputs
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                       # React + TypeScript + Tailwind
│   ├── src/
│   │   ├── App.tsx                 # Root router
│   │   ├── pages/
│   │   │   ├── UploadPage.tsx      # Drag-drop upload + demo mode
│   │   │   └── DashboardPage.tsx   # Full analytics dashboard
│   │   ├── types/index.ts          # TypeScript interfaces
│   │   └── utils/api.ts            # API client + polling
│   ├── tailwind.config.js
│   └── Dockerfile
│
└── docker-compose.yml
```

---

## ML Pipeline

### Features (20 engineered from CV)

| Category | Features |
|---|---|
| Joint angles | Elbow angle at release, min elbow angle, knee bend, wrist snap, trunk lean, arm extension |
| Ball flight | Arc angle, release velocity, peak velocity, trajectory smoothness, release height |
| Timing | Shot duration, jump height, release timing at peak, gather-to-release rhythm |
| Spatial | Player distance to hoop, shot angle, lateral balance, follow-through score |

### Models

- **Form Scorer** — XGBoost regressor → predicts form score 0–100
- **Shot Predictor** — XGBoost classifier → predicts make probability 0–1

Both fall back to weighted rule-based scoring before training data is collected.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- (Optional) CUDA GPU for faster inference

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train models with synthetic data (no video dataset needed to start)
python scripts/train_models.py --synthetic

# Start API server
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/api/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Docker (full stack)

```bash
docker compose up --build
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

---

## Training Your Own Models

### Step 1 — Collect annotated shots

Create `data/raw/annotations.json`:
```json
{
  "shots": [
    {
      "video_path": "data/raw/videos/shot_001.mp4",
      "made_shot": 1,
      "form_score_label": 82.0,
      "player_id": "player_01"
    }
  ]
}
```

### Step 2 — Preprocess videos → features

```bash
python scripts/preprocess_dataset.py \
  --annotations data/raw/annotations.json \
  --output data/processed/shots.csv
```

### Step 3 — Train models

```bash
python scripts/train_models.py \
  --data data/processed/shots.csv \
  --output data/models/
```

### Step 4 — Evaluate

```bash
python scripts/evaluate_models.py \
  --models data/models/ \
  --data data/processed/shots.csv \
  --output data/models/eval/
```

The API will automatically hot-reload the new models on the next request.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/analysis/upload` | Upload video, get session_id |
| `GET`  | `/api/v1/analysis/status/{id}` | Poll for results |
| `POST` | `/api/v1/analysis/demo` | Run demo (no upload) |
| `GET`  | `/api/v1/models/info` | Model registry info |
| `GET`  | `/api/v1/health` | Health check |

### Example Response

```json
{
  "session_id": "abc123...",
  "status": "completed",
  "result": {
    "form_score": 76.4,
    "form_grade": "Good",
    "make_probability": 0.621,
    "confidence": 0.88,
    "features": {
      "elbow_angle_release": 94.2,
      "ball_arc_angle": 47.3,
      "release_timing_score": 0.82,
      ...
    },
    "feedback": [
      {
        "aspect": "Shot arc angle",
        "rating": "good",
        "message": "Excellent shot arc — ideal angle for maximizing scoring window",
        "metric_value": 47.3,
        "benchmark_range": "38–55"
      }
    ],
    "ball_trajectory": [...],
    "pose_frames": [...],
    "model_version": "1.0.0"
  }
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Tailwind CSS, Recharts, Framer Motion |
| Backend | FastAPI, Pydantic v2, Uvicorn |
| CV | OpenCV, MediaPipe Pose, YOLOv8 (Ultralytics) |
| ML | NumPy, Pandas, Scikit-learn, XGBoost, SciPy |
| Infra | Docker, Docker Compose, Nginx |

---

## Roadmap

- [ ] Real-time webcam analysis via WebRTC
- [ ] Multi-shot session history and trend tracking
- [ ] YOLOv8 custom-trained hoop detection head
- [ ] TensorFlow/PyTorch LSTM for temporal sequence modelling
- [ ] Player comparison dashboard
- [ ] Shot chart heatmap overlay
- [ ] Mobile app (React Native + on-device TFLite)

---

## Project by Shanvi — [cookiesandcoding.in](https://cookiesandcoding.in)

Built with FastAPI · MediaPipe · YOLOv8 · XGBoost · React
