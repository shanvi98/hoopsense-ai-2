"""
HoopSense AI — Model Registry & Versioning
Tracks model versions, metadata, and evaluation metrics
"""
import os
import json
import pickle
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

REGISTRY_FILE = os.path.join(settings.MODEL_DIR, "registry.json")


def load_registry() -> Dict[str, Any]:
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"models": [], "active_version": None}


def register_model(
    version: str,
    form_mae: float,
    form_r2: float,
    success_accuracy: float,
    success_auc: float,
    n_train_samples: int,
    notes: str = "",
) -> None:
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    registry = load_registry()
    entry = {
        "version": version,
        "registered_at": datetime.utcnow().isoformat(),
        "metrics": {
            "form_mae": round(form_mae, 3),
            "form_r2": round(form_r2, 3),
            "success_accuracy": round(success_accuracy, 3),
            "success_auc": round(success_auc, 3),
        },
        "n_train_samples": n_train_samples,
        "notes": notes,
        "artifacts": {
            "form_model":    f"form_model_{version}.pkl",
            "success_model": f"success_model_{version}.pkl",
            "scaler":        f"scaler_{version}.pkl",
        },
    }
    registry["models"].append(entry)
    registry["active_version"] = version
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)
    logger.info("Model v%s registered in registry", version)


def get_active_version() -> Optional[str]:
    registry = load_registry()
    return registry.get("active_version")


def list_versions() -> list:
    return load_registry().get("models", [])
