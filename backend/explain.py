"""Look inside a trained model: its weights, its probability curve, and why it
predicted one match the way it did.

The model is a pipeline: fill missing values, standardize, then softmax
logistic regression. Each outcome k gets a score z_k = b_k + sum(w_k * x),
where x is a standardized input. Only differences between scores matter, so
the home-win score minus the away-win score is

    z_H - z_A = (b_H - b_A) + sum((w_H - w_A) * x)  =  log(P(home) / P(away)).

That difference is what the "toward home" numbers here measure: positive
pushes the prediction toward a home win, negative toward an away win.

From other code:
    from explain import weight_table, elo_curve, contributions
"""

import re

import numpy as np
import pandas as pd

from elo import expected_score

SIDES = {"home": "Home", "away": "Away"}
NAMES = {
    "ppg": "points per game",
    "gf": "goals scored per game",
    "ga": "goals conceded per game",
    "sotf": "shots on target per game",
    "sota": "shots on target conceded per game",
    "vs_exp": "results vs Elo expectation",
}
MATCH_LABELS = {
    "elo_diff": "Elo gap (home minus away)",
    "elo_sum": "Elo level (both ratings added)",
    "elo_expected": "Elo expected home score",
    "home_advantage": "League home advantage",
    "championship": "Championship game",
}


def label(feature):
    """A readable name: 'home_sotf_5' -> 'Home shots on target per game, last 5'."""
    if feature in MATCH_LABELS:
        return MATCH_LABELS[feature]
    side, _, rest = feature.partition("_")
    if side not in SIDES:
        return feature
    who = SIDES.get(side, side)
    if rest == "venue_ppg_5":
        return f"{who} team's {side} points per game, last 5"
    if rest == "rest_days":
        return f"{who} rest days"
    if rest == "came_up":
        return f"{who} just promoted"
    match = re.fullmatch(r"(.+)_(\d+)", rest)
    if match and match.group(1) in NAMES:
        return f"{who} {NAMES[match.group(1)]}, last {match.group(2)}"
    return feature


def _parts(model):
    """The pipeline's imputer, scaler and logistic regression."""
    return model[0], model[1], model[-1]


def _home_away(lr):
    classes = list(lr.classes_)
    return classes.index("H"), classes.index("A")


def weight_table(model, features):
    """One row per input: its weights for home, draw and away, and home minus away.

    A weight is the change in that outcome's score when the input goes up
    by one standard deviation. Sorted by how much the input separates a
    home win from an away win.
    """
    lr = _parts(model)[2]
    table = pd.DataFrame(lr.coef_.T, index=features, columns=lr.classes_)[["H", "D", "A"]]
    table["toward_home"] = table["H"] - table["A"]
    table.insert(0, "label", [label(f) for f in features])
    return table.reindex(table["toward_home"].abs().sort_values(ascending=False).index)


def elo_curve(model, features, home_advantage, gaps=np.arange(-400, 401, 10)):
    """Home, draw and away probability as the Elo gap changes, other inputs at their average.

    The average is the training rows' mean of each input. elo_expected is
    worked out from each gap and `home_advantage`, as Elo itself would, so
    the two Elo inputs stay consistent.
    """
    imputer, scaler, lr = _parts(model)
    rows = pd.DataFrame(np.tile(scaler.mean_, (len(gaps), 1)), columns=features)
    rows["elo_diff"] = gaps
    if "elo_expected" in features:
        rows["elo_expected"] = [expected_score(g, 0, home_advantage) for g in gaps]
    probs = model.predict_proba(rows[features])
    order = [list(lr.classes_).index(o) for o in ("H", "D", "A")]
    probs = probs[:, order]
    return pd.DataFrame({"gap": gaps, "home": probs[:, 0], "draw": probs[:, 1],
                         "away": probs[:, 2]})


def contributions(model, features, row):
    """Why one match got its prediction: each input's push toward a home or away win.

    `row` is a one-row DataFrame with the model's inputs. Returns the
    contributions, sorted by size, and the intercept b_H - b_A. The
    intercept plus all contributions equals log(P(home) / P(away)).
    """
    imputer, scaler, lr = _parts(model)
    filled = imputer.transform(row[features])
    x = scaler.transform(filled)[0]
    h, a = _home_away(lr)
    push = (lr.coef_[h] - lr.coef_[a]) * x
    table = pd.DataFrame({"feature": features, "label": [label(f) for f in features],
                          "value": filled[0], "standardized": x, "toward_home": push})
    table = table.reindex(table["toward_home"].abs().sort_values(ascending=False).index)
    return table.reset_index(drop=True), float(lr.intercept_[h] - lr.intercept_[a])
