"""
Data layer. The recommender core only needs list[str].
Swap any of these in — or write your own — and nothing else changes.
"""

from __future__ import annotations




'''def load_sample() -> list[str]:
    """Tiny built-in dataset so the repo runs with zero setup."""
    return [
        "The central bank raised interest rates to curb inflation.",
        "Stock markets fell after the new monetary policy announcement.",
        "The team trained a transformer model on a large text corpus.",
        "New GPU architecture speeds up deep learning training.",
        "The striker scored a hat-trick in the championship final.",
        "The national team qualified for the World Cup playoffs.",
        "A new vaccine showed strong results in phase three trials.",
        "Doctors recommend regular screening for early diagnosis.",
    ]
'''

def load_from_csv(path: str, text_column: str) -> list[str]:
    """Load documents from a CSV column."""
    import pandas as pd
    df = pd.read_csv(path)
    return df[text_column].dropna().astype(str).tolist()


def load_20newsgroups(n: int = 2000) -> list[str]:
    """
    Real, well-known dataset (~18k news posts). Good for a portfolio demo.
    pip install scikit-learn
    """
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    return [t for t in data.data[:n] if t.strip()]
    
def load_sample() -> list[str]:
    return [
        "NASA launched a new probe to study the rings of Saturn.",          # 0 космос
        "Astronomers discovered an exoplanet orbiting a distant star.",     # 1 космос
        "The rocket successfully reached orbit after a smooth launch.",     # 2 космос
        "To make pasta carbonara you need eggs, cheese and guanciale.",     # 3 готовка
        "Slowly caramelize the onions before adding them to the soup.",     # 4 готовка
        "This chocolate cake recipe uses dark cocoa and fresh butter.",     # 5 готовка
        "The team scored a last-minute goal to win the championship.",      # 6 спорт
        "He broke the world record in the 100 meter sprint.",              # 7 спорт
        "The tennis final went to five sets before a dramatic finish.",     # 8 спорт
        "The marathon runner collapsed just before the finish line.",       # 9 спорт
    ]
