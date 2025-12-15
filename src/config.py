# Project configuration

SEED = 42

# Dataset
DATASET_NAME = "ag_news"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

# Limits for faster experiments
MAX_TRAIN_SAMPLES = 20000
MAX_TEST_SAMPLES = 5000
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
