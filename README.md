# Content-Based Document Recommender

A lightweight, data-agnostic recommender that suggests relevant documents from
just the **first few clicks** — no user history or behaviour log required.

## Problem

Classic collaborative-filtering recommenders need a large interaction log and
suffer from the *cold-start* problem: they can't recommend anything to a new
user or for a fresh catalog. This project solves recommendation from the very
first interaction using document content alone.

## Approach

```
documents (list[str])
    → encoder (sentence-transformers)   → normalized vectors
    → KMeans clustering                 → topic map (for inspection / viz)
    → cosine similarity to the centroid of clicked docs → recommendations
```

The user profile is simply the mean vector of the documents they clicked — the
"direction" of their interest. Recommendations are the nearest unseen documents
to that direction. It works from the **first click**.

The core (`recommender.py`) only depends on a `list[str]`, so it works on any
text corpus — news, product descriptions, papers, reviews — by swapping the
data loader in `data_sources.py`. Nothing else changes.

## Tech stack

- Python
- sentence-transformers (embeddings)
- scikit-learn (KMeans, cosine similarity)
- numpy

## Quick start

```bash
pip install -r requirements.txt
python recommender.py
```

This runs a self-contained demo on a small built-in dataset (no downloads
beyond the encoder model).

## Use your own data

```python
from recommender import DocRecommender
from data_sources import load_from_csv

docs = load_from_csv("reviews.csv", text_column="text")
rec = DocRecommender(n_clusters=8).fit(docs)

# user clicked documents 3 and 7 → recommend 5 more
print(rec.recommend([3, 7], top_k=5))
```

For non-English text, use a multilingual encoder:

```python
DocRecommender(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

## Notes

- Embeddings are L2-normalized, so dot product equals cosine similarity.
- Clustering is used for the topic map / visualization; recommendations come
  from nearest-neighbour search to the click centroid, which is more robust
  than hard cluster assignment.
