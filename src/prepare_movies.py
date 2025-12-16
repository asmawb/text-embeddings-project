import ast
import json
import pandas as pd

from .config import (
    RAW_MOVIES_CSV,
    PROCESSED_CSV,
    DATA_PROCESSED_DIR,
    GENRES,
    MIN_SAMPLES_PER_GENRE,
)


def _parse_genres_cell(cell):
    """
    Parse TMDB 'genres' cell:
    usually like: '[{"id": 28, "name": "Action"}, ...]'
    """
    if pd.isna(cell):
        return []

    s = str(cell).strip()
    if not s or s == "[]":
        return []

    # Prefer strict JSON, fallback to Python-literal parsing (some exports vary)
    items = None
    try:
        items = json.loads(s)
    except Exception:
        try:
            items = ast.literal_eval(s)
        except Exception:
            return []

    if not isinstance(items, list):
        return []

    out = []
    for x in items:
        if isinstance(x, dict) and "name" in x:
            out.append(str(x["name"]))
    return out


def _pick_one_genre(genres_list):
    """
    Convert multi-genre movies into single-label classification by priority order.
    """
    gset = set(genres_list)
    for g in GENRES:
        if g in gset:
            return g
    return None


def main():
    df = pd.read_csv(RAW_MOVIES_CSV, low_memory=False)

    required = ["overview", "genres"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {RAW_MOVIES_CSV}. "
            f"Found: {list(df.columns)}"
        )

    df = df[["overview", "genres"]].copy()
    df["overview"] = df["overview"].fillna("").astype(str).str.strip()
    df = df[df["overview"].str.len() > 0].copy()

    df["genres_list"] = df["genres"].apply(_parse_genres_cell)
    df["label"] = df["genres_list"].apply(_pick_one_genre)
    df = df.dropna(subset=["label"]).copy()

    out = df[["overview", "label"]].rename(columns={"overview": "text"})

    # remove labels with too-few samples 
    counts = out["label"].value_counts()
    keep = counts[counts >= MIN_SAMPLES_PER_GENRE].index
    out = out[out["label"].isin(keep)].copy()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED_CSV, index=False)

    print("Saved:", PROCESSED_CSV)
    print("Rows:", len(out))
    print("Label distribution:")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
