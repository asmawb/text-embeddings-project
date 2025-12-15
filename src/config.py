from pathlib import Path

# ---------- Paths ----------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Dataset ----------
# We'll use HuggingFace "datasets" to download Kaggle "the-movies-dataset" (movies_metadata.csv)
HF_DATASET = "wykonos/movies"

RAW_CSV_NAME = "movies_metadata.csv"
RAW_CSV_PATH = RAW_DIR / RAW_CSV_NAME

# After processing
PROCESSED_CSV_PATH = PROCESSED_DIR / "movies_5genres.csv"

# Target genres (exact labels must match the dataset's "genres" names)
TARGET_GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance"]

# How many movies per genre to keep (balances dataset)
PER_GENRE = 2000  # adjust if your machine is slow (e.g., 1000)

# ---------- Models ----------
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Saved artifacts
TFIDF_ARTIFACT = METRICS_DIR / "tfidf_model.joblib"
SBERT_ARTIFACT = METRICS_DIR / "sbert_model.joblib"

# Saved metrics
METRICS_JSON = METRICS_DIR / "movie_genre_metrics.json"
REPORTS_TXT = METRICS_DIR / "movie_genre_reports.txt"
