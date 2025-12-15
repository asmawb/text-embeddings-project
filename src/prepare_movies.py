import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from .config import (
    HF_DATASET,
    RAW_CSV_PATH,
    PROCESSED_CSV_PATH,
    TARGET_GENRES,
    PER_GENRE,
)

def _safe_json_load(x: str):
    try:
        return json.loads(x)
    except Exception:
        return None

def _extract_primary_genre(genres_field, allowed):
    """
    genres_field in Kaggle movies_metadata is a stringified JSON list like:
    '[{"id": 28, "name": "Action"}, ...]'
    We map each movie to a single primary genre among allowed.
    """
    if genres_field is None or not isinstance(genres_field, str):
        return None

    g = _safe_json_load(genres_field)
    if not isinstance(g, list):
        return None

    names = [d.get("name") for d in g if isinstance(d, dict) and "name" in d]
    names = [n for n in names if isinstance(n, str)]
    for n in names:
        if n in allowed:
            return n
    return None

def _clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    print("Downloading dataset from HuggingFace datasets...")
    ds = load_dataset(HF_DATASET)

    # The HF dataset usually exposes a split like "train" only.
    # Convert to pandas.
    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    # Try to find movies_metadata.csv-like columns.
    # Expect: overview, genres
    if "overview" not in df.columns or "genres" not in df.columns:
        raise ValueError(f"Expected columns 'overview' and 'genres'. Found: {df.columns.tolist()[:30]}")

    # Save raw CSV (optional but helpful)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Saved raw CSV to: {RAW_CSV_PATH}")

    # Keep only needed columns
    df = df[["overview", "genres"]].copy()
    df["overview"] = df["overview"].apply(_clean_text)
    df = df[df["overview"].str.len() >= 30]

    # Extract a single label
    tqdm.pandas(desc="Extracting primary genre")
    df["label"] = df["genres"].progress_apply(lambda x: _extract_primary_genre(x, set(TARGET_GENRES)))
    df = df.dropna(subset=["label"]).copy()

    # Balance dataset: PER_GENRE per class
    buckets = defaultdict(list)
    for idx, row in df.iterrows():
        buckets[row["label"]].append(row)

    rows = []
    for g in TARGET_GENRES:
        items = buckets.get(g, [])
        if len(items) < 200:
            raise ValueError(f"Not enough samples for genre '{g}'. Found {len(items)}")
        np.random.shuffle(items)
        rows.extend(items[:PER_GENRE])

    out = pd.DataFrame(rows)[["overview", "label"]].sample(frac=1.0, random_state=42).reset_index(drop=True)

    PROCESSED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_CSV_PATH, index=False)
    print(f"Saved processed dataset: {PROCESSED_CSV_PATH}")
    print("Class counts:")
    print(out["label"].value_counts())

if __name__ == "__main__":
    main()
