import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .config import SBERT_MODEL

def make_tfidf(X_train, X_test, max_features=50000, ngram_range=(1, 2)):
    """
    TF-IDF baseline embeddings (sparse matrices).
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        lowercase=True,
    )
    Z_train = vec.fit_transform(X_train)
    Z_test = vec.transform(X_test)
    return vec, Z_train, Z_test

def make_sbert(X_train, X_test, batch_size=64):
    """
    Sentence-BERT embeddings (dense numpy arrays).
    Normalized so dot-product == cosine similarity.
    """
    model = SentenceTransformer(SBERT_MODEL)
    Z_train = model.encode(
        list(X_train),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    Z_test = model.encode(
        list(X_test),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    return model, Z_train, Z_test
