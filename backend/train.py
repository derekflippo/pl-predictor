"""The model: how it is built, tuned and trained.

The model is a multinomial (softmax) logistic regression. Missing values are
filled with the training median, then every feature is standardized. Both
steps are fitted on the training rows only.

Seasons (only Premier League games are ever predicted and scored):
    2002-03 onward       training rows
    2015-16 to 2018-19   validation: tuning and every choice in this file
    2019-20 to 2025-26   test: scored only by evaluate.py, after the settings
                         in model_settings.json are frozen

Walk-forward: to predict a season, the model trains on every season from
2002-03 up to the one before it. Validating 2017-18 trains on 2002-03 to
2016-17. A model never trains on the season it predicts, or a later one.

Usage:
    python backend/train.py --tune   # step 4 search; writes backend/model_settings.json
    python backend/train.py          # trains on every finished match; prints the weights
"""

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS
from load_data import DATA_DIR

TRAIN_FROM = "2002-03"
VALIDATE = ["2015-16", "2016-17", "2017-18", "2018-19"]
TEST = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
OUTCOMES = ["H", "D", "A"]  # column order of every probability array in this project

LEFT_OUT = [
    "home_came_down", "away_came_down",  # always 0 in Premier League rows
    # Almost always 5 and 10, so a short window is 8 to 11 SD from the mean
    # after standardizing, and the model extrapolates wildly (tested 2026-11-16).
    "home_games_5", "away_games_5", "home_games_10", "away_games_10",
]
# home_elo and away_elo go in as their difference and their sum: the same
# information, but the weights say "gap" and "level" directly.
RAW_ELO = ["home_elo", "away_elo"]
ELO_FEATURES = ["elo_diff", "elo_sum", "elo_expected", "home_advantage"]
FORM_FEATURES = [c for c in FEATURE_COLUMNS
                 if c not in LEFT_OUT + RAW_ELO + ["elo_expected", "home_advantage"]]
MODEL_FEATURES = ELO_FEATURES + FORM_FEATURES
ELO_ONLY = ["elo_diff"]  # the baseline: logistic regression on the rating difference


def _both(*names):
    return [f"{side}_{n}" for n in names for side in ("home", "away")]


# Feature groups for the ablations. Every model feature except elo_diff is in one.
GROUPS = {
    "Elo curve (elo_expected)": ["elo_expected"],
    "Elo level (elo_sum)": ["elo_sum"],
    "home advantage": ["home_advantage"],
    "points (ppg)": _both("ppg_5", "ppg_10"),
    "goals (gf, ga)": _both("gf_5", "gf_10", "ga_5", "ga_10"),
    "shots on target (sotf, sota)": _both("sotf_5", "sotf_10", "sota_5", "sota_10"),
    "vs Elo expectation (vs_exp)": _both("vs_exp_5", "vs_exp_10"),
    "venue points (venue_ppg_5)": _both("venue_ppg_5"),
    "rest days": _both("rest_days"),
    "promoted (came_up)": _both("came_up"),
}
C_GRID = [float(f"{10 ** e:.3g}") for e in np.arange(-4, 0.01, 0.25)]  # 0.0001 to 1
ODDS = ["odds_home", "odds_draw", "odds_away"]
KEY = ["league", "season", "home_team", "away_team"]
SETTINGS_PATH = Path(__file__).resolve().parent / "model_settings.json"
# scikit-learn 1.8 deprecated LogisticRegression's `penalty`: l1_ratio alone
# picks the mix (0 is L2, 1 is L1). Older versions need penalty="elasticnet".
NEW_PENALTY_API = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DEFAULT_SETTINGS = {"features": MODEL_FEATURES, "c": 1.0, "l1_ratio": None,
                    "championship_rows": False}


# --- Data ---------------------------------------------------------------------------

def load_table(data_dir=DATA_DIR):
    """Feature rows for both leagues, with the derived Elo columns and the pre-match odds."""
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["date"])
    odds = pd.read_csv(data_dir / "matches.csv", usecols=KEY + ODDS)
    table = features.merge(odds, on=KEY, how="left", validate="one_to_one")
    return add_derived(table)


def add_derived(table):
    """elo_diff and elo_sum from the two ratings, and a 0/1 Championship flag."""
    table = table.copy()
    table["elo_diff"] = table["home_elo"] - table["away_elo"]
    table["elo_sum"] = table["home_elo"] + table["away_elo"]
    table["championship"] = (table["league"] == "E1").astype(float)
    return table


def before_test(table):
    """Drop the test seasons and everything after, so tuning can't see them."""
    return table[table["season"] < TEST[0]]


def season_rows(table, season):
    """The Premier League games of one season: the rows a walk-forward step predicts."""
    return table[(table["season"] == season) & (table["league"] == "E0")]


def walk_forward(table, seasons, fit_predict):
    """Predict each season with a model trained only on the seasons before it.

    fit_predict(train, rows) returns probabilities for `rows`. `train` holds
    both leagues from TRAIN_FROM to the season before; fit_predict picks the
    rows it wants. Returns the predicted rows and their probabilities.
    """
    parts, probs = [], []
    for season in seasons:
        train = table[(table["season"] >= TRAIN_FROM) & (table["season"] < season)]
        rows = season_rows(table, season)
        probs.append(fit_predict(train, rows))
        parts.append(rows)
    return pd.concat(parts, ignore_index=True), np.vstack(probs)


# --- Model --------------------------------------------------------------------------

def make_model(c=1.0, l1_ratio=None):
    """Fill missing values, standardize, then softmax logistic regression.

    l1_ratio None is the L2 penalty. 1 is L1 (lasso), and between 0 and 1 is
    elastic net, a mix of the two. C = 1/λ, so a smaller C is a stronger penalty.
    """
    if l1_ratio is None:
        lr = LogisticRegression(C=c, max_iter=1000)
    elif NEW_PENALTY_API:
        lr = LogisticRegression(C=c, l1_ratio=l1_ratio, solver="saga", max_iter=20000)
    else:
        lr = LogisticRegression(C=c, penalty="elasticnet", l1_ratio=l1_ratio,
                                solver="saga", max_iter=20000)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), lr)


def fit(rows, features=MODEL_FEATURES, c=1.0, l1_ratio=None):
    """Train on `rows`. Fails if the optimizer stops before reaching the minimum."""
    model = make_model(c, l1_ratio)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(rows[features], rows["result"])
    return model


def predict(model, rows, features=MODEL_FEATURES):
    """Probabilities with columns in OUTCOMES order: home, draw, away."""
    probs = model.predict_proba(rows[features])
    order = [list(model.classes_).index(o) for o in OUTCOMES]
    return probs[:, order]


def model_features(settings):
    """The settings' features, plus the Championship flag when Championship rows train too."""
    return list(settings["features"]) + (["championship"] if settings["championship_rows"] else [])


def training_rows(train, settings):
    return train if settings["championship_rows"] else train[train["league"] == "E0"]


def model_predictor(settings):
    """A fit_predict for walk_forward that trains the model with these settings."""
    features = model_features(settings)

    def fit_predict(train, rows):
        model = fit(training_rows(train, settings), features, settings["c"], settings["l1_ratio"])
        return predict(model, rows, features)
    return fit_predict


def elo_only_predictor(train, rows):
    train = train[train["league"] == "E0"]
    return predict(fit(train, ELO_ONLY), rows, ELO_ONLY)


def base_rate_predictor(train, rows):
    return base_rates(train[train["league"] == "E0"], len(rows))


def bookmaker_predictor(train, rows):
    return bookmaker_probs(rows)


def train_final(table, settings):
    """The production model: trained on every finished match from TRAIN_FROM."""
    rows = table[(table["season"] >= TRAIN_FROM) & table["result"].notna()]
    return fit(training_rows(rows, settings), model_features(settings),
               settings["c"], settings["l1_ratio"])


def load_settings(path=SETTINGS_PATH):
    return json.loads(Path(path).read_text())


# --- Baselines and scores -----------------------------------------------------------

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


# --- Tuning (step 4) ----------------------------------------------------------------

def validation_losses(table, settings):
    """Per-game log loss on the validation seasons, walk-forward."""
    rows, probs = walk_forward(before_test(table), VALIDATE, model_predictor(settings))
    return match_scores(rows["result"], probs)["log_loss"].to_numpy()


def one_se_choice(losses_by_c):
    """The one-standard-error rule: the smallest C within one SE of the best C.

    losses_by_c maps each C to its per-game losses on the same games. The
    SE is that of the per-game difference from the best C, so the choice is
    the strongest penalty that is not clearly worse than the best.
    Returns (chosen C, best C).
    """
    best = min(losses_by_c, key=lambda c: np.mean(losses_by_c[c]))
    for c in sorted(losses_by_c):
        diff, se = paired(losses_by_c[c], losses_by_c[best])
        if diff <= se:
            return c, best


def clearly_better(new, old):
    """True when `new` has lower loss than `old` by more than one standard error."""
    diff, se = paired(new, old)
    return diff < -se


def tune(table, log=print):
    """Step 4 on the validation seasons. Returns the settings and a record of every result.

    Rules, written before the search:
      - C: the one-SE rule over C_GRID.
      - The default is the L2 model on MODEL_FEATURES, Premier League rows only.
        A change (L1 or elastic net, Championship rows, dropping a feature
        group) is kept only when it is better by more than one SE.
    """
    table = before_test(table)
    record = {"rule": tune.__doc__.split("Rules, written before the search:")[1].strip()}

    def sweep(settings, grid=C_GRID):
        losses = {c: validation_losses(table, {**settings, "c": c}) for c in grid}
        chosen, best = one_se_choice(losses)
        log(f"  chosen C {chosen} (best {best}): {np.mean(losses[chosen]):.4f} "
            f"(best {np.mean(losses[best]):.4f})")
        grid_record = {str(c): round(float(np.mean(v)), 5) for c, v in losses.items()}
        return {**settings, "c": chosen}, losses[chosen], {"chosen": chosen, "best": best,
                                                           "grid": grid_record}

    def compare(new, old):
        diff, se = paired(new, old)
        return {"log_loss": round(float(np.mean(new)), 5),
                "minus_current": [round(diff, 5), round(se, 5)]}

    log("1. C for the L2 model on every feature")
    settings, current, record["l2"] = sweep(DEFAULT_SETTINGS)

    log("2. L1 and elastic net")
    coarse = C_GRID[::2]
    for name, ratio in (("l1", 1.0), ("elastic_net", 0.5)):
        candidate, losses, record[name] = sweep({**settings, "l1_ratio": ratio}, coarse)
        record[name].update(compare(losses, current))
        if clearly_better(losses, current):
            settings, current = candidate, losses
            record[name]["kept"] = True

    log("3. Championship rows")
    candidate, losses, record["championship_rows"] = sweep({**settings, "championship_rows": True})
    record["championship_rows"].update(compare(losses, current))
    if clearly_better(losses, current):
        settings, current = candidate, losses
        record["championship_rows"]["kept"] = True

    log("4. Ablations: drop one feature group at a time")
    record["ablations"] = {}
    dropped = []
    for name, group in GROUPS.items():
        kept = [f for f in settings["features"] if f not in group]
        losses = validation_losses(table, {**settings, "features": kept})
        record["ablations"][name] = compare(losses, current)
        diff, se = record["ablations"][name]["minus_current"]
        log(f"  without {name:<30} {diff:+.4f} ± {se:.4f}")
        if clearly_better(losses, current):
            dropped.append(name)
    if dropped:
        features = [f for f in settings["features"]
                    if not any(f in GROUPS[name] for name in dropped)]
        log(f"  dropping {dropped}; C again")
        settings, current, record["after_drops"] = sweep({**settings, "features": features})
    record["dropped_groups"] = dropped

    log("5. For reference: Elo features only, and the Elo-only baseline")
    elo = {**settings, "features": [f for f in settings["features"] if f in ELO_FEATURES]}
    record["elo_features_only"] = compare(validation_losses(table, elo), current)
    rows, probs = walk_forward(table, VALIDATE, elo_only_predictor)
    record["elo_only_baseline"] = compare(
        match_scores(rows["result"], probs)["log_loss"].to_numpy(), current)
    record["final_validation_log_loss"] = round(float(np.mean(current)), 5)
    return settings, record


def save_settings(settings, record, path=SETTINGS_PATH):
    out = {**settings, "tuning": {"validation_seasons": f"{VALIDATE[0]} to {VALIDATE[-1]}",
                                  "games": 380 * len(VALIDATE), "tuned_on": str(date.today()),
                                  **record}}
    Path(path).write_text(json.dumps(out, indent=2) + "\n")
    return path


# --- Command line -------------------------------------------------------------------

def show_weights(model, features):
    w = weights(model, features)
    print(f"  {'intercept':<22}" + "".join(f"{v:>+8.3f}" for v in intercepts(model)))
    for name in (w["H"] - w["A"]).abs().sort_values(ascending=False).index:
        print(f"  {name:<22}" + "".join(f"{v:>+8.3f}" for v in w.loc[name]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tune", action="store_true", help="run step 4 and save the settings")
    args = parser.parse_args()
    table = load_table()

    if args.tune:
        settings, record = tune(table)
        print(f"\nSettings: C {settings['c']}, l1_ratio {settings['l1_ratio']}, "
              f"Championship rows {settings['championship_rows']}, "
              f"{len(settings['features'])} features")
        print(f"Saved {save_settings(settings, record)}")
        return

    settings = load_settings()
    features = model_features(settings)
    model = train_final(table, settings)
    rows = table[(table["season"] >= TRAIN_FROM) & table["result"].notna()]
    rows = training_rows(rows, settings)
    print(f"Trained on {len(rows)} matches, {rows['season'].min()} to {rows['season'].max()}. "
          f"C {settings['c']}, {len(features)} features.")
    print("Weights on standardized features (z for home / draw / away):")
    show_weights(model, features)


if __name__ == "__main__":
    main()
