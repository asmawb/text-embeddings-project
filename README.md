# Text Embeddings for News Classification (AG News)

## Goal
Compare classic TF-IDF embeddings vs modern Sentence-BERT (SBERT) embeddings for text classification.

We train a Logistic Regression classifier on:
1) TF-IDF (bag-of-words baseline)
2) SBERT embeddings (dense semantic vectors)
3) SBERT + PCA(128) (dimensionality reduction)

## Dataset
- AG News (4 classes)

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
