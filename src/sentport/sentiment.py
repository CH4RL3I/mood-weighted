"""FinBERT scoring (cached) and aggregation to per-stock daily sentiment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from .timing import available_date

MODEL_ID = "ProsusAI/finbert"
Scorer = Callable[[list[str]], np.ndarray]  # texts -> (n, 3) probs [positive, negative, neutral]


def finbert_scorer(batch_size: int = 64, threads: int = 4, max_length: int = 64) -> Scorer:
    """Return a scorer that runs ProsusAI/finbert on CPU."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    labels = {v.lower(): k for k, v in model.config.id2label.items()}
    order = [labels["positive"], labels["negative"], labels["neutral"]]

    def score(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i : i + batch_size], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")  # fmt: skip
            with torch.inference_mode():
                p = torch.softmax(model(**enc).logits, dim=-1).numpy()
            out.append(p[:, order])
        return np.vstack(out) if out else np.zeros((0, 3))

    return score


def score_headlines(
    headlines: pd.DataFrame,
    scorer: Scorer,
    cache: Path,
    chunk: int = 2048,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Score unique headline texts once; results are cached to ``cache`` (parquet).

    Returns a frame with columns text, p_pos, p_neg, p_neu, score (= p_pos - p_neg).
    """
    texts = pd.Index(headlines["headline"].astype(str).unique())
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(
        columns=["text", "p_pos", "p_neg", "p_neu"])  # fmt: skip
    todo = texts.difference(pd.Index(done["text"]))
    log(f"{len(texts)} unique headlines, {len(done)} cached, {len(todo)} to score")
    parts = [done]
    for i in range(0, len(todo), chunk):
        batch = list(todo[i : i + chunk])
        p = scorer(batch)
        parts.append(pd.DataFrame({"text": batch, "p_pos": p[:, 0], "p_neg": p[:, 1],
                                   "p_neu": p[:, 2]}))  # fmt: skip
        pd.concat(parts, ignore_index=True).to_parquet(cache)
        log(f"  scored {min(i + chunk, len(todo))}/{len(todo)}")
    res = pd.concat(parts, ignore_index=True).drop_duplicates("text")
    res["score"] = res["p_pos"] - res["p_neg"]
    return res


def aggregate_daily(headlines: pd.DataFrame, scores: pd.DataFrame, lag_days: int = 0) -> pd.DataFrame:
    """Per stock and publication date: headline count and score sum, plus the availability date.

    Output columns: ticker, date, n, score_sum, avail_date. No headline text.
    """
    m = headlines.merge(scores[["text", "score"]], left_on="headline", right_on="text", how="inner")
    g = m.groupby(["ticker", "date"]).agg(n=("score", "size"), score_sum=("score", "sum"))
    g = g.reset_index()
    g["avail_date"] = available_date(g["date"], date_only=True, lag_days=lag_days)
    return g
