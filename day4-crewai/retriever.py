"""
A from-scratch vector retriever. No Chroma, no FAISS, no LangChain.

The entire vector database is three pieces:
  1. A matrix of embeddings (one row per document)
  2. A similarity function (cosine)
  3. A top-k selector (sort, take top k indices)

Once you've built this, every "vector database" you encounter is just
this + persistence + a speed-optimized index. Nothing more.

This version uses sentence-transformers for LOCAL embeddings — no API,
no cost, runs on your CPU.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_PATH = "data/corpus_index.npz"
META_PATH = "data/corpus_meta.csv"

print(f"Loading local embedding model: {EMBED_MODEL_NAME}")
_model = SentenceTransformer(EMBED_MODEL_NAME)
print("  Ready.")


# ---------- Step 1: Embed texts locally ----------
def embed_texts(texts: list[str], input_type: str = "document") -> np.ndarray:
    """
    Embed a list of texts using a local sentence-transformer model.
    Returns (N, EMBED_DIM) array.

    input_type is ignored (this model doesn't distinguish doc vs query),
    but kept in the signature for API compatibility.
    """
    embeddings = _model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


# ---------- Step 2: Cosine similarity (the math, by hand) ----------
def cosine_similarity(query_vec: np.ndarray, corpus_matrix: np.ndarray) -> np.ndarray:
    """
    cosine(a, b) = (a . b) / (||a|| * ||b||)
    Returns (N,) array of similarities, one per corpus document.
    """
    dots = corpus_matrix @ query_vec
    query_norm = np.linalg.norm(query_vec)
    corpus_norms = np.linalg.norm(corpus_matrix, axis=1)
    denom = query_norm * corpus_norms + 1e-10
    return dots / denom


# ---------- Step 3: Top-k selector ----------
def top_k(similarities: np.ndarray, k: int) -> list[int]:
    """Return indices of the k highest similarities, sorted descending."""
    return np.argsort(similarities)[-k:][::-1].tolist()


# ---------- Step 4: Build & persist the index ----------
def build_index(corpus_csv: str = "data/reviews_corpus.csv") -> None:
    """Embed the entire corpus once and save to disk. Run this once."""
    print(f"Loading corpus from {corpus_csv}...")
    df = pd.read_csv(corpus_csv)
    print(f"  {len(df)} documents")

    print("Embedding corpus (using local model)...")
    embeddings = embed_texts(df["text"].tolist(), input_type="document")

    print(f"Saving index to {INDEX_PATH}...")
    np.savez(INDEX_PATH, embeddings=embeddings, review_ids=df["review_id"].values)
    df.to_csv(META_PATH, index=False)
    print(f"  Done. Shape: {embeddings.shape}")


# ---------- Step 5: Load + search interface ----------
class CorpusRetriever:
    """
    Loads the prebuilt index from disk and exposes a search method.
    This is the object the agent's tool function will use.
    """

    def __init__(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"No index at {INDEX_PATH}. Run build_index() first."
            )
        data = np.load(INDEX_PATH, allow_pickle=True)
        self.embeddings = data["embeddings"]
        self.review_ids = data["review_ids"]
        self.meta = pd.read_csv(META_PATH).set_index("review_id")
        print(f"Loaded index: {self.embeddings.shape}")

    def search(self, query_text: str, k: int = 3,
               filter_label: str | None = None) -> list[dict]:
        """
        Return the top-k most similar reviews to query_text.

        filter_label: if 'fake' or 'genuine', only return matches with that
        label. Used in Day 2's metadata-filtering experiment.
        """
        query_vec = embed_texts([query_text], input_type="query")[0]
        sims = cosine_similarity(query_vec, self.embeddings)

        if filter_label is not None:
            mask = self.meta.loc[self.review_ids]["label"].values == filter_label
            sims = np.where(mask, sims, -np.inf)

        top_indices = top_k(sims, k)

        results = []
        for idx in top_indices:
            rid = self.review_ids[idx]
            row = self.meta.loc[rid]
            results.append({
                "review_id": str(rid),
                "similarity": float(sims[idx]),
                "label": row["label"],
                "rating": int(row["rating"]),
                "text": row["text"][:200],
            })
        return results


# ---------- Run as script: builds the index ----------
if __name__ == "__main__":
    build_index()