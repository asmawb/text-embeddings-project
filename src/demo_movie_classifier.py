import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import TFIDF_ARTIFACT, SBERT_ARTIFACT

def topk(probs, genres, k=3):
    idx = np.argsort(-probs)[:k]
    return [(genres[i], float(probs[i])) for i in idx]

def demo_tfidf(text):
    pack = joblib.load(TFIDF_ARTIFACT)
    vec = pack["vectorizer"]
    clf = pack["classifier"]
    genres = pack["genres"]

    Z = vec.transform([text])
    probs = clf.predict_proba(Z)[0]
    pred = genres[int(np.argmax(probs))]
    return pred, topk(probs, genres, k=3)

def demo_sbert(text):
    pack = joblib.load(SBERT_ARTIFACT)
    clf = pack["classifier"]
    sbert_name = pack["sbert_model"]
    genres = pack["genres"]

    model = SentenceTransformer(sbert_name)
    emb = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    probs = clf.predict_proba(emb)[0]
    pred = genres[int(np.argmax(probs))]
    return pred, topk(probs, genres, k=3)

def main():
    print("Movie Genre Classifier Demo (TF-IDF vs SBERT)")
    print("Type a movie plot/description and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        text = input("Plot> ").strip()
        if text.lower() in ["q", "quit", "exit"]:
            break
        if len(text) < 20:
            print("Please type a longer description.\n")
            continue

        pred1, top1 = demo_tfidf(text)
        pred2, top2 = demo_sbert(text)

        print("\nTF-IDF prediction:", pred1)
        print("Top-3:", top1)
        print("\nSBERT prediction:", pred2)
        print("Top-3:", top2)
        print()

if __name__ == "__main__":
    main()
