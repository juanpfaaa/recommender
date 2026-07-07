"""
Content-based document recommender.

Pipeline:
    documents (list[str])
        -> encoder (sentence-transformers)  -> vectors
        -> clustering (KMeans)              -> topic map  [optional, for viz]
        -> recommend by cosine similarity to the centroid of clicked docs

The core is data-agnostic: it takes a plain list of strings.
Plug any data source by returning list[str] (see data_sources.py).
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class DocRecommender:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_clusters: int = 8):
        """
        model_name : any sentence-transformers model.
                     use a multilingual one (e.g. 'paraphrase-multilingual-MiniLM-L12-v2')
                     if your documents are not in English.
        n_clusters : number of topic clusters (only used for the topic map).
        """
        self.model = SentenceTransformer(model_name)
        self.n_clusters = n_clusters

        self.documents: list[str] = []
        self.vectors: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # 1. Build the index
    # ------------------------------------------------------------------ #
    def fit(self, documents: list[str]) -> "DocRecommender":
        """Encode all documents and build the topic map."""
        self.documents = documents
        self.vectors = self.model.encode(
            documents,
            normalize_embeddings=True,   # so dot product == cosine similarity
            show_progress_bar=True,
        )
        self.labels = KMeans(
            n_clusters=self.n_clusters, random_state=42, n_init="auto"
        ).fit_predict(self.vectors)
        return self

    # ------------------------------------------------------------------ #
    # 2. Recommend
    # ------------------------------------------------------------------ #
    def recommend(self, clicked_idx: list[int], top_k: int = 5) -> list[int]:
        """
        Given the indices of documents the user clicked, return top_k
        recommended document indices (nearest to the clicks' centroid).

        Works from the very first click — no behaviour log required.
        """
        if self.vectors is None:
            raise RuntimeError("call fit() first")
        if not clicked_idx:
            raise ValueError("need at least one clicked document")

        # user profile = mean of clicked vectors (the 'direction' of interest)
        profile = self.vectors[clicked_idx].mean(axis=0, keepdims=True)

        sims = cosine_similarity(profile, self.vectors)[0]
        sims[clicked_idx] = -np.inf          # don't recommend what was clicked

        return np.argsort(sims)[::-1][:top_k].tolist()

    # ------------------------------------------------------------------ #
    # 3. Helpers for inspection / demo
    # ------------------------------------------------------------------ #
    def cluster_of(self, idx: int) -> int:
        return int(self.labels[idx])

    def preview(self, idx: int, n_chars: int = 100) -> str:
        return self.documents[idx][:n_chars].replace("\n", " ")


if __name__ == "__main__":
    # tiny self-contained demo (no external data needed)
    from data_sources import load_sample

    docs = load_sample()
    rec = DocRecommender(n_clusters=4).fit(docs)

    clicks = [0, 1]  # pretend the user clicked the first two docs
    print("Clicked:")
    for i in clicks:
        print(f"  [{rec.cluster_of(i)}] {rec.preview(i)}")

    print("\nRecommended:")
    for i in rec.recommend(clicks, top_k=3):
        print(f"  [{rec.cluster_of(i)}] {rec.preview(i)}")
