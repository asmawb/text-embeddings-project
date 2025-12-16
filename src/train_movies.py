import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers import SentenceTransformer

from .config import (
    PROCESSED_CSV,
    MODELS_DIR,
    METRICS_DIR,
    GENRES,
    TEST_SIZE,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_STOP_WORDS,
    SBERT_MODEL,
    LOGREG_MAX_ITER,
)

def train_tfidf_logreg(Xtr, Xte, ytr, yte):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            stop_words=TFIDF_STOP_WORDS,
        )),
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER)),
    ])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    return pipe, {"acc": acc, "f1_macro": f1}

def train_sbert_logreg(Xtr, Xte, ytr, yte):
    embedder = SentenceTransformer(SBERT_MODEL)
    Etr = embedder.encode(list(Xtr), show_progress_bar=True, convert_to_numpy=True)
    Ete = embedder.encode(list(Xte), show_progress_bar=True, convert_to_numpy=True)

    clf = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    clf.fit(Etr, ytr)

    pred = clf.predict(Ete)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred, average="macro")
    return clf, {"acc": acc, "f1_macro": f1}

def main():
    df = pd.read_csv(PROCESSED_CSV)
    if len(df) < 50:
        raise ValueError(f"Processed dataset too small ({len(df)} rows). Run prepare_movies.py and check output.")

    X = df["text"].astype(str).values
    y = df["label"].astype(str).values

    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # Train TF-IDF model (pipeline works directly on text)
    tfidf_pipe, tfidf_metrics = train_tfidf_logreg(Xtr, Xte, ytr, yte)
    tfidf_path = MODELS_DIR / "tfidf_logreg.joblib"
    joblib.dump(tfidf_pipe, tfidf_path)

    # Train SBERT model (classifier ONLY; demo must embed text)
    sbert_clf, sbert_metrics = train_sbert_logreg(Xtr, Xte, ytr, yte)
    sbert_path = MODELS_DIR / "sbert_logreg.joblib"
    joblib.dump(sbert_clf, sbert_path)

    # Choose best by macro F1
    best_name = "sbert" if sbert_metrics["f1_macro"] >= tfidf_metrics["f1_macro"] else "tfidf"

    # Save a tiny “best model meta” so demo knows how to run it
    best_meta = {
        "best_model": best_name,
        "labels": GENRES,
        "tfidf_path": str(tfidf_path),
        "sbert_path": str(sbert_path),
        "sbert_model_name": SBERT_MODEL,
    }
    best_meta_path = MODELS_DIR / "best_model_meta.json"
    with open(best_meta_path, "w") as f:
        json.dump(best_meta, f, indent=2)

    metrics_out = {
        "tfidf": tfidf_metrics,
        "sbert": sbert_metrics,
        "best_model": best_name,
    }
    metrics_path = METRICS_DIR / "movie_genre_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)

    print("Saved models to:", MODELS_DIR)
    print("Saved metrics to:", metrics_path)
    print("Best model:", best_name)

if __name__ == "__main__":
    main()
