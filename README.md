# Movie Genre Classification from Plot Descriptions (Text Embeddings)

## Goal
Build a text classification system that predicts a movie’s genre
(Action, Comedy, Drama, Horror, Romance) using only its plot description.

The project compares traditional TF-IDF embeddings with modern
Sentence-BERT (SBERT) semantic embeddings.

---

## Dataset
**The Movies Dataset** (Kaggle – movies_metadata.csv)

- Text input: movie plot description (`overview`)
- Labels: movie genre
- 5 target genres: Action, Comedy, Drama, Horror, Romance
- Movies with multiple genres are mapped to a single primary genre
- Dataset is balanced across genres

Raw data is not committed to GitHub.

---

## Methods

### Embeddings
1. **TF-IDF**
   - Bag-of-words with uni-grams and bi-grams
   - Captures keyword frequency

2. **Sentence-BERT (SBERT)**
   - Pretrained semantic text embeddings
   - Encodes meaning beyond exact words

### Classifier
- Logistic Regression
- Same classifier used for all embeddings to ensure fair comparison

---

## Training & Evaluation
- Train / test split: 80% / 20%
- Metrics:
  - Accuracy
  - Macro F1-score (handles class balance fairly)

---

## Results

| Model | Accuracy | Macro F1 |
|------|----------|----------|
| TF-IDF + Logistic Regression | 0.623 | 0.511 |
| **SBERT + Logistic Regression** | **0.632** | **0.515** |

---

## Discussion
Movie genre classification from plot descriptions is a challenging task
because genres often overlap and plot summaries may be ambiguous.

TF-IDF relies on explicit keywords, which limits its performance when
genre-specific words are not present.

SBERT embeddings capture semantic information and improve performance
slightly over TF-IDF, demonstrating the benefit of semantic embeddings
for complex text classification tasks.

---

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.prepare_movies
python -m src.train_movies
