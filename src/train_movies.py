import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from .config import (
    PROCESSED_CSV_PATH,
    TARGET_GENRES,
    SBERT_MODEL,
    TFIDF_ARTIFACT,
    SBERT_ARTIFACT,
    METRICS_JSON,
    REPORTS_TXT,
)

def load_data():
    if not PROCESSED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {PROCESSED_CSV_PATH}\n"
            f"Run: python -m src.prepare_movies"
        )

    df = pd.read_csv(PROCESSED_CSV_PATH)
    df = df.dropna(subset=["overview", "label"])
    X = df["overview"].astype(str).values
    y = df["label"].astype(str).values

    # enforce label set
    keep = np.isin(y, TARGET_GENRES)
    X, y = X[keep], y[keep]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return Xtr, Xte, ytr, yte

def eval_metrics(y_true, y_pred, name):
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    metrics = {"model": name, "accuracy": acc, "macro_f1": macro}
    report = classification_report(y_true, y_pred, digits=4)
    return metrics, report

def run_tfidf_logreg(Xtr, Xte, ytr, yte):
    vec = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    Ztr = vec.fit_transform(Xtr)
    Zte = vec.transform(Xte)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)

    metrics, report = eval_metrics(yte, pred, "tfidf_logreg")
    return metrics, report, vec, clf

def run_sbert_logreg(Xtr, Xte, ytr, yte):
    model = SentenceTransformer(SBERT_MODEL)

    Ztr = model.encode(
        list(Xtr),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    Zte = model.encode(
        list(Xte),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Ztr, ytr)
    pred = clf.predict(Zte)

    metrics, report = eval_metrics(yte, pred, "sbert_logreg")
    return metrics, report, clf

def main():
    Xtr, Xte, ytr, yte = load_data()

    m1, r1, tfidf_vec, tfidf_clf = run_tfidf_logreg(Xtr, Xte, ytr, yte)
    m2, r2, sbert_clf = run_sbert_logreg(Xtr, Xte, ytr, yte)

    all_metrics = [m1, m2]

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_TXT.parent.mkdir(parents=True, exist_ok=True)

    with open(METRICS_JSON, "w") as f:
        json.dump(all_metrics, f, indent=2)

    with open(REPORTS_TXT, "w") as f:
        f.write("=== TF-IDF + Logistic Regression ===\n")
        f.write(r1 + "\n\n")
        f.write("=== SBERT + Logistic Regression ===\n")
        f.write(r2 + "\n")

    # Save artifacts for demo
    joblib.dump(
        {"vectorizer": tfidf_vec, "classifier": tfidf_clf, "genres": TARGET_GENRES},
        TFIDF_ARTIFACT
    )
    joblib.dump(
        {"classifier": sbert_clf, "sbert_model": SBERT_MODEL, "genres": TARGET_GENRES},
        SBERT_ARTIFACT
    )

    print("Saved metrics:")
    print(f"- {METRICS_JSON}")
    print(f"- {REPORTS_TXT}")
    print("Saved model artifacts:")
    print(f"- {TFIDF_ARTIFACT}")
    print(f"- {SBERT_ARTIFACT}")
    print("\nMetrics:")
    print(json.dumps(all_metrics, indent=2))

if __name__ == "__main__":
    main()
