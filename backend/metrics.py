"""Scores for probability predictions, and the two baselines that need no model.

Every probability array in this project has three columns in OUTCOMES
order: home win, draw, away win.

From other code:
    from metrics import match_scores, summarize, paired
"""

import numpy as np
import pandas as pd

OUTCOMES = ["H", "D", "A"]
ODDS = ["odds_home", "odds_draw", "odds_away"]


def base_rates(train, n):
    """The training rows' home/draw/away shares, the same for every game."""
    shares = train["result"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    return np.tile(shares, (n, 1))


def bookmaker_probs(rows):
    """Implied probabilities 1/odds, scaled so the three add up to 1.

    The scaling removes the bookmaker's margin: the raw 1/odds add up to
    a little more than 1. Raises if a game has no odds, rather than
    letting NaN spread into every score.
    """
    odds = rows[ODDS].to_numpy(dtype=float)
    missing = np.isnan(odds).any(axis=1)
    if missing.any():
        names = [c for c in ("season", "home_team", "away_team") if c in rows.columns]
        games = rows.loc[missing, names].astype(str).agg(" ".join, axis=1).tolist()
        raise ValueError(f"No bookmaker odds for {missing.sum()} game(s): {games[:5]}")
    implied = 1 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def match_scores(results, probs):
    """Per-match log loss and Brier score, and whether the favourite won.

    Brier score here is the sum over the three outcomes of
    (probability - 1 if it happened else 0) squared: 0 is perfect, 2 the worst.
    """
    actual = np.array([[r == o for o in OUTCOMES] for r in results], dtype=float)
    return pd.DataFrame({
        "log_loss": -np.log((probs * actual).sum(axis=1)),
        "brier": ((probs - actual) ** 2).sum(axis=1),
        "correct": probs.argmax(axis=1) == actual.argmax(axis=1),
    })


def summarize(scores):
    return {"log_loss": float(scores["log_loss"].mean()),
            "brier": float(scores["brier"].mean()),
            "accuracy": float(scores["correct"].mean())}


def paired(a, b):
    """Average per-game difference a - b, and its standard error.

    Both arrays score the same games, so the game-to-game noise they share
    cancels out, and the standard error is that of the differences.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def calibration(results, probs, edges=np.linspace(0, 1, 11)):
    """For each outcome, predictions grouped by probability, with how often it happened."""
    rows = []
    results = np.asarray(results)
    for i, outcome in enumerate(OUTCOMES):
        p = probs[:, i]
        happened = results == outcome
        group = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
        for g in np.unique(group):
            inside = group == g
            rate = happened[inside].mean()
            rows.append({"outcome": outcome, "low": float(edges[g]), "high": float(edges[g + 1]),
                         "games": int(inside.sum()), "predicted": float(p[inside].mean()),
                         "actual": float(rate),
                         "standard_error": float(np.sqrt(rate * (1 - rate) / inside.sum()))})
    return pd.DataFrame(rows)
