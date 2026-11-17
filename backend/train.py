"""Train the first model and check it on the validation seasons.

The first model is a multinomial (softmax) logistic regression on every
feature except came_down and games_5/games_10 (see LEFT_OUT), with an L2
penalty at scikit-learn's default C = 1. Missing values are filled with the training median, then every
feature is standardized. Both steps are fitted on the training rows only.

Seasons (Premier League rows only):
    2002-03 to 2014-15   train
    2015-16 to 2018-19   validate: every comparison here is on these
    2019-20 to 2025-26   test: sealed, never scored here

Usage:
    python backend/train.py   # prints a report, writes backend/data/first_model.json
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from load_data import DATA_DIR

TRAIN = ("2002-03", "2014-15")
VALIDATE = ("2015-16", "2018-19")
OUTCOMES = ["H", "D", "A"]  # column order of every probability array in this file
LEFT_OUT = [
    "home_came_down", "away_came_down",  # always 0 in Premier League rows
    # Almost always 5 and 10, so a short window is 8 to 11 SD from the mean
    # after standardizing, and the model extrapolates wildly (tested 2026-11-16).
    "home_games_5", "away_games_5", "home_games_10", "away_games_10",
]
MODEL_FEATURES = [c for c in FEATURE_COLUMNS if c not in LEFT_OUT]
ELO_ONLY = ["elo_expected"]
ODDS = ["odds_home", "odds_draw", "odds_away"]
KEY = ["league", "season", "home_team", "away_team"]
REPORT_PATH = DATA_DIR / "first_model.json"


def load_table(data_dir=DATA_DIR):
    """Premier League feature rows, with the pre-match odds attached for the benchmark."""
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["date"])
    odds = pd.read_csv(data_dir / "matches.csv", usecols=KEY + ODDS)
    table = features.merge(odds, on=KEY, how="left", validate="one_to_one")
    return table[table["league"] == "E0"].reset_index(drop=True)


def split(table):
    """Training rows and validation rows. Test seasons are never returned."""
    train = table[table["season"].between(*TRAIN)]
    validate = table[table["season"].between(*VALIDATE)]
    return train.reset_index(drop=True), validate.reset_index(drop=True)


def make_model(c=1.0):
    """Fill missing values, standardize, then softmax logistic regression with L2.

    C = 1/λ, so a smaller C means a stronger penalty.
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c, max_iter=1000),
    )


def fit(rows, features=MODEL_FEATURES, c=1.0):
    """Train on `rows`. Fails if the optimizer stops before reaching the minimum."""
    model = make_model(c)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(rows[features], rows["result"])
    return model


def predict(model, rows, features=MODEL_FEATURES):
    """Probabilities with columns in OUTCOMES order: home, draw, away."""
    probs = model.predict_proba(rows[features])
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return probs[:, order]


def base_rates(train, n):
    """The training seasons' home/draw/away shares, the same for every game."""
    shares = train["result"].value_counts(normalize=True).reindex(OUTCOMES).to_numpy()
    return np.tile(shares, (n, 1))


def bookmaker_probs(rows):
    """Implied probabilities 1/odds, scaled so the three add up to 1.

    The scaling removes the bookmaker's margin: the raw 1/odds add up to
    a little more than 1.
    """
    implied = 1 / rows[ODDS].to_numpy(dtype=float)
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
    """Average per-match difference a - b, and its standard error."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def weights(model, features=MODEL_FEATURES):
    """One weight per feature and outcome, on standardized features.

    A weight is how much that outcome's score z changes when the feature
    goes up by one standard deviation. Only differences between the three
    outcomes matter to the probabilities.
    """
    lr = model[-1]
    table = pd.DataFrame(lr.coef_.T, index=features, columns=lr.classes_)
    return table[OUTCOMES]


def intercepts(model):
    lr = model[-1]
    return pd.Series(lr.intercept_, index=lr.classes_)[OUTCOMES]


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


def evaluate(table):
    """Train the first model and the baselines, and score them on the validation seasons."""
    train, validate = split(table)
    model = fit(train)
    elo_only = fit(train, ELO_ONLY)

    probs = {
        "model": predict(model, validate),
        "elo_only": predict(elo_only, validate, ELO_ONLY),
        "base_rates": base_rates(train, len(validate)),
        "bookmakers": bookmaker_probs(validate),
    }
    scores = {name: match_scores(validate["result"], p) for name, p in probs.items()}
    train_probs = predict(model, train)
    train_scores = match_scores(train["result"], train_probs)
    train_base = match_scores(train["result"], base_rates(train, len(train)))

    comparisons = {}
    for name, s in scores.items():
        row = summarize(s)
        if name != "model":
            row["model_minus_this"] = paired(scores["model"]["log_loss"], s["log_loss"])
        comparisons[name] = row
    comparisons["always_home"] = {"accuracy": float((validate["result"] == "H").mean())}

    return {
        "model": model, "train": train, "validate": validate, "probs": probs,
        "train_probs": train_probs, "comparisons": comparisons,
        "train_log_loss": summarize(train_scores)["log_loss"],
        "train_base_log_loss": summarize(train_base)["log_loss"],
        "weights": weights(model), "intercepts": intercepts(model),
        "calibration": calibration(validate["result"], probs["model"]),
        "missing_filled": {"train": int(train[MODEL_FEATURES].isna().sum().sum()),
                           "validate": int(validate[MODEL_FEATURES].isna().sum().sum())},
    }


def report(r):
    train, validate = r["train"], r["validate"]
    print(f"Training rows:   {len(train)} ({TRAIN[0]} to {TRAIN[1]})")
    print(f"Validation rows: {len(validate)} ({VALIDATE[0]} to {VALIDATE[1]})")
    print(f"Features: {len(MODEL_FEATURES)} (left out: {', '.join(LEFT_OUT)})")
    print(f"Missing values filled: {r['missing_filled']['train']} in training, "
          f"{r['missing_filled']['validate']} in validation\n")

    print("Step 2a. Does it beat doing nothing? (training seasons)")
    print(f"  Model log loss {r['train_log_loss']:.4f}, "
          f"average shares {r['train_base_log_loss']:.4f}\n")

    print("Step 2b. Weights on standardized features (z for home / draw / away)")
    print(f"  {'intercept':<20}" + "".join(f"{v:>+8.3f}" for v in r["intercepts"]))
    w = r["weights"]
    order = (w["H"] - w["A"]).abs().sort_values(ascending=False).index
    for name in order:
        print(f"  {name:<20}" + "".join(f"{v:>+8.3f}" for v in w.loc[name]))

    p = r["probs"]["model"]
    print("\nStep 2c. Do the predictions look sane? (validation seasons)")
    print("  Average predicted: " + "  ".join(f"{o} {v:.1%}" for o, v in zip(OUTCOMES, p.mean(0))))
    actual = validate["result"].value_counts(normalize=True).reindex(OUTCOMES)
    print("  Actual shares:     " + "  ".join(f"{o} {v:.1%}" for o, v in actual.items()))
    print(f"  Highest draw probability: {p[:, 1].max():.1%}. "
          f"Games where a draw is the favourite: {(p.argmax(1) == 1).sum()}")

    print(f"\nStep 3a. Validation scores ({len(validate)} games)")
    print(f"  {'':<12}{'log loss':>9}{'accuracy':>10}{'Brier':>8}   model minus this (log loss)")
    for name, row in r["comparisons"].items():
        if "log_loss" not in row:
            print(f"  {name:<12}{'-':>9}{row['accuracy']:>10.1%}{'-':>8}")
            continue
        diff = row.get("model_minus_this")
        extra = f"   {diff[0]:+.4f} ± {diff[1]:.4f}" if diff else ""
        print(f"  {name:<12}{row['log_loss']:>9.4f}{row['accuracy']:>10.1%}"
              f"{row['brier']:>8.4f}{extra}")

    gap = r["comparisons"]["model"]["log_loss"] - r["train_log_loss"]
    print(f"\nStep 3b. Validation minus training log loss: {gap:+.4f}")

    print("\nStep 3c. Calibration (validation): predicted vs happened")
    for _, row in r["calibration"].iterrows():
        if row["games"] >= 20:
            print(f"  {row['outcome']} {row['low']:.0%}-{row['high']:.0%}: {row['games']:>4} games, "
                  f"predicted {row['predicted']:.1%}, happened {row['actual']:.1%} "
                  f"± {row['standard_error']:.1%}")


def save(r, path=REPORT_PATH):
    """Write the results, including every validation prediction, as JSON."""
    v = r["validate"]
    games = pd.DataFrame({
        "season": v["season"], "date": v["date"].dt.strftime("%Y-%m-%d"),
        "home_team": v["home_team"], "away_team": v["away_team"],
        "home_goals": v["home_goals"].astype(int), "away_goals": v["away_goals"].astype(int),
        "result": v["result"], "home_elo": v["home_elo"].round(1),
        "away_elo": v["away_elo"].round(1),
    })
    for name in ("model", "elo_only", "bookmakers"):
        games[name] = [[round(float(x), 4) for x in row] for row in r["probs"][name]]
    out = {
        "seasons": {"train": TRAIN, "validate": VALIDATE},
        "features": MODEL_FEATURES, "left_out": LEFT_OUT,
        "rows": {"train": len(r["train"]), "validate": len(v)},
        "train_log_loss": r["train_log_loss"], "train_base_log_loss": r["train_base_log_loss"],
        "comparisons": r["comparisons"],
        "weights": r["weights"].round(4).to_dict(orient="index"),
        # one standard deviation of each feature in the training rows: the step a weight is per
        "feature_sd": dict(zip(MODEL_FEATURES, r["model"][1].scale_.round(4).tolist())),
        "intercepts": r["intercepts"].round(4).to_dict(),
        "calibration": r["calibration"].to_dict(orient="records"),
        "games": games.to_dict(orient="records"),
    }
    path.write_text(json.dumps(out, indent=1))
    return path


def main():
    r = evaluate(load_table())
    report(r)
    print(f"\nSaved {save(r)}")


if __name__ == "__main__":
    main()
