import json
from functools import lru_cache

import joblib
import numpy as np

from .config import MODELS_DIR, SBERT_MODEL


@lru_cache(maxsize=1)
def _load_best_meta() -> dict:
    """Load the model selection metadata written by src.train_movies."""
    meta_path = MODELS_DIR / "best_model_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {meta_path}. Run:\n"
            f"  python -m src.prepare_movies\n"
            f"  python -m src.train_movies\n"
        )
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _get_sbert_embedder():
    """Create the SBERT embedder once (expensive) and reuse it."""
    from sentence_transformers import SentenceTransformer

    meta = _load_best_meta()
    name = meta.get("sbert_model_name", SBERT_MODEL)
    return SentenceTransformer(name)


def load_best_model():
    """
    Load and return the best saved model (TF-IDF pipeline or SBERT classifier).
    The choice and paths are stored in outputs/models/best_model_meta.json.
    """
    meta = _load_best_meta()
    best = meta.get("best_model")

    if best == "tfidf":
        return joblib.load(meta["tfidf_path"])

    if best == "sbert":
        return joblib.load(meta["sbert_path"])

    raise ValueError(f"Unknown best_model='{best}' in best_model_meta.json")


def predict_one(model, text: str):
    """
    Predict a single genre from a movie overview.

    Returns:
        (predicted_label, confidence)
        confidence is the max predicted probability when available; otherwise None.
    """
    meta = _load_best_meta()

    text = "" if text is None else str(text).strip()
    if not text:
        raise ValueError("Please type a non-empty description.")

    best = meta.get("best_model")

    # TF-IDF pipeline: text -> prediction directly
    if best == "tfidf":
        X = [text]
        pred = model.predict(X)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            conf = float(np.max(proba))
        return pred, conf

    # SBERT classifier: text -> embedding -> prediction
    if best == "sbert":
        embedder = _get_sbert_embedder()
        emb = embedder.encode([text], convert_to_numpy=True)  # shape (1, dim)

        pred = model.predict(emb)[0]
        conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(emb)[0]
            conf = float(np.max(proba))
        return pred, conf

    raise ValueError(f"Unknown best_model='{best}' in best_model_meta.json")
