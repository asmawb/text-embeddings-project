from pathlib import Path

# ======================================================
# Paths
# ======================================================
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"

OUTPUTS = REPO_ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
METRICS_DIR = OUTPUTS / "metrics"

# TMDB 5000 dataset file (from Kaggle)
RAW_MOVIES_CSV = DATA_RAW / "tmdb_5000_movies.csv"

# Processed dataset you generate
PROCESSED_CSV = DATA_PROCESSED / "movies_6genres.csv"

# ======================================================
# Labels (6 genres)
# ======================================================
GENRES = [
    "Drama",
    "Comedy",
    "Action",
    "Horror",
    "Thriller",
    "Romance",
]

# ======================================================
# Train / Test split
# ======================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ======================================================
# TF-IDF settings
# ======================================================
TFIDF_MAX_FEATURES = 50000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_STOP_WORDS = "english"

# ======================================================
# SBERT settings
# ======================================================
SBERT_MODEL = "all-MiniLM-L6-v2"

# ======================================================
# Logistic Regression settings
# ======================================================
LOGREG_MAX_ITER = 2000
