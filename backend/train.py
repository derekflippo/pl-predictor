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
from metrics import ODDS, OUTCOMES, match_scores, paired

TRAIN_FROM = "2002-03"
VALIDATE = ["2015-16", "2016-17", "2017-18", "2018-19"]
TEST = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

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
ELO_ONLY_SETTINGS = {"features": ELO_ONLY, "c": 1.0, "l1_ratio": None,
                     "championship_rows": False}


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
KEY = ["league", "season", "home_team", "away_team"]
SETTINGS_PATH = Path(__file__).resolve().parent / "model_settings.json"
# scikit-learn 1.8 deprecated LogisticRegression's `penalty`: l1_ratio alone
# picks the mix (0 is L2, 1 is L1). Older versions need penalty="elasticnet".
NEW_PENALTY_API = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
# saga, the solver elastic net needs, visits the rows in a random order. A
# fixed seed makes every run give exactly the same model (without it, runs
# differed by about 0.000006 in probability).
SEED = 0
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
        lr = LogisticRegression(C=c, l1_ratio=l1_ratio, solver="saga", max_iter=20000,
                                random_state=SEED)
    else:
        lr = LogisticRegression(C=c, penalty="elasticnet", l1_ratio=l1_ratio,
                                solver="saga", max_iter=20000, random_state=SEED)
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


def train_final(table, settings):
    """The production model: trained on every finished match from TRAIN_FROM."""
    rows = table[(table["season"] >= TRAIN_FROM) & table["result"].notna()]
    return fit(training_rows(rows, settings), model_features(settings),
               settings["c"], settings["l1_ratio"])


def load_settings(path=SETTINGS_PATH):
    return json.loads(Path(path).read_text())


# --- Inspecting a model -------------------------------------------------------------

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


TUNING_RULES = """\
- C: the one-SE rule over C_GRID.
- The default is the L2 model on MODEL_FEATURES, Premier League rows only.
  A change (L1 or elastic net, Championship rows, dropping a feature
  group) is kept only when it is better by more than one SE.
- Feature groups: a group is a candidate when removing it alone is better
  by more than one SE. Candidates are then dropped one at a time, best
  first, and each drop is kept only if it is still better by more than one
  SE than the features left at that point."""


def tune(table, log=print, score=validation_losses):
    """Step 4 on the validation seasons. Returns the settings and a record of every result.

    The rules are in TUNING_RULES, written before the search ran. `score`
    returns per-game validation losses for a settings dict; tests pass a
    stand-in.
    """
    table = before_test(table)
    record = {"rule": TUNING_RULES}

    def sweep(settings, grid=C_GRID):
        losses = {c: score(table, {**settings, "c": c}) for c in grid}
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
    candidates = []
    for name, group in GROUPS.items():
        kept = [f for f in settings["features"] if f not in group]
        losses = score(table, {**settings, "features": kept})
        record["ablations"][name] = compare(losses, current)
        diff, se = record["ablations"][name]["minus_current"]
        log(f"  without {name:<30} {diff:+.4f} ± {se:.4f}")
        if clearly_better(losses, current):
            candidates.append((float(np.mean(losses)), name))
    # Two groups can each look removable only because the other covers for
    # it, so drop them one at a time and check each against what's left.
    dropped, features = [], list(settings["features"])
    for _, name in sorted(candidates):
        kept = [f for f in features if f not in GROUPS[name]]
        losses = score(table, {**settings, "features": kept})
        if clearly_better(losses, current):
            features, current = kept, losses
            dropped.append(name)
        else:
            log(f"  keeping {name}: not better once {dropped} are gone")
    if dropped:
        log(f"  dropped {dropped}; C again")
        settings, current, record["after_drops"] = sweep({**settings, "features": features})
    record["drop_candidates"] = [name for _, name in sorted(candidates)]
    record["dropped_groups"] = dropped

    log("5. For reference: Elo features only, and the Elo-only baseline")
    elo = {**settings, "features": [f for f in settings["features"] if f in ELO_FEATURES]}
    record["elo_features_only"] = compare(score(table, elo), current)
    record["elo_only_baseline"] = compare(score(table, ELO_ONLY_SETTINGS), current)
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
