"""Score the model season by season, next to the baselines.

The final test: for each season from 2019-20 to 2025-26, the model trains
on every season from 2002-03 up to the one before it, with the settings
frozen in model_settings.json, and predicts that season. Those settings
were chosen on the validation seasons only, so the test seasons are games
the whole system has never seen.

Baselines, trained the same walk-forward way:
    Elo only         logistic regression on the rating difference
    Average shares   the training seasons' home/draw/away shares for every game
    Bookmakers       1/odds, scaled to remove the margin (the benchmark)
    Always home      accuracy only

Usage:
    python backend/evaluate.py               # test seasons; writes backend/data/evaluation.json
    python backend/evaluate.py --validation  # the same report on the validation seasons
"""

import argparse
import json

import numpy as np
import pandas as pd

from load_data import DATA_DIR
from train import (OUTCOMES, TEST, VALIDATE, base_rate_predictor, before_test,
                   bookmaker_predictor, calibration, elo_only_predictor, load_settings,
                   load_table, match_scores, model_predictor, paired, summarize, walk_forward)

NAMES = {"model": "Model", "elo_only": "Elo only", "base_rates": "Average shares",
         "bookmakers": "Bookmakers"}
RESULTS_PATH = DATA_DIR / "evaluation.json"


def compare(table, seasons, settings):
    """Walk-forward predictions and scores for the model and every baseline."""
    predictors = {"model": model_predictor(settings), "elo_only": elo_only_predictor,
                  "base_rates": base_rate_predictor, "bookmakers": bookmaker_predictor}
    probs = {}
    for name, fit_predict in predictors.items():
        rows, probs[name] = walk_forward(table, seasons, fit_predict)
    scores = {name: match_scores(rows["result"], p) for name, p in probs.items()}

    overall = {}
    for name, s in scores.items():
        overall[name] = summarize(s)
        if name != "model":
            overall[name]["model_minus_this"] = paired(scores["model"]["log_loss"], s["log_loss"])
    overall["always_home"] = {"accuracy": float((rows["result"] == "H").mean())}

    per_season = []
    for season in seasons:
        inside = (rows["season"] == season).to_numpy()
        entry = {"season": season, "games": int(inside.sum()),
                 "always_home": float((rows["result"][inside] == "H").mean())}
        for name, s in scores.items():
            entry[name] = summarize(s[inside])
        entry["model_minus_bookmakers"] = paired(scores["model"]["log_loss"][inside],
                                                 scores["bookmakers"]["log_loss"][inside])
        per_season.append(entry)

    return {"seasons": seasons, "rows": rows, "probs": probs, "scores": scores,
            "overall": overall, "per_season": per_season,
            "calibration": {name: calibration(rows["result"], probs[name])
                            for name in ("model", "bookmakers")}}


def report(r):
    seasons = r["seasons"]
    print(f"Walk-forward, {seasons[0]} to {seasons[-1]}: {len(r['rows'])} games\n")
    print("Log loss by season (lower is better)")
    print(f"  {'season':<9}" + "".join(f"{NAMES[n]:>16}" for n in NAMES) + f"{'model - bookies':>20}")
    for e in r["per_season"]:
        diff, se = e["model_minus_bookmakers"]
        print(f"  {e['season']:<9}" + "".join(f"{e[n]['log_loss']:>16.4f}" for n in NAMES)
              + f"{diff:>+13.4f} ± {se:.4f}")

    print("\nAll seasons")
    print(f"  {'':<16}{'log loss':>9}{'accuracy':>10}{'Brier':>8}   model minus this (log loss)")
    for name, row in r["overall"].items():
        label = NAMES.get(name, "Always home")
        if "log_loss" not in row:
            print(f"  {label:<16}{'-':>9}{row['accuracy']:>10.1%}{'-':>8}")
            continue
        diff = row.get("model_minus_this")
        extra = f"   {diff[0]:+.4f} ± {diff[1]:.4f}" if diff else ""
        print(f"  {label:<16}{row['log_loss']:>9.4f}{row['accuracy']:>10.1%}"
              f"{row['brier']:>8.4f}{extra}")

    print("\nCalibration of the model: predicted vs happened (groups of 40+ games)")
    for _, row in r["calibration"]["model"].iterrows():
        if row["games"] >= 40:
            off = (row["actual"] - row["predicted"]) / row["standard_error"]
            print(f"  {row['outcome']} {row['low']:.0%}-{row['high']:.0%}: {row['games']:>4} games, "
                  f"predicted {row['predicted']:.1%}, happened {row['actual']:.1%} "
                  f"± {row['standard_error']:.1%} ({off:+.1f} SE)")


def markdown_table(r):
    """The README's results table."""
    lines = ["| Season | Model | Elo only | Average shares | Bookmakers | Model accuracy | Bookmaker accuracy |",
             "|---|---|---|---|---|---|---|"]
    for e in r["per_season"] + [{"season": "**All**", **r["overall"]}]:
        lines.append(f"| {e['season']} | " + " | ".join(f"{e[n]['log_loss']:.3f}" for n in NAMES)
                     + f" | {e['model']['accuracy']:.1%} | {e['bookmakers']['accuracy']:.1%} |")
    return "\n".join(lines)


def save(r, settings, path=RESULTS_PATH):
    """Write the results, including every game's probabilities, as JSON."""
    rows = r["rows"]
    games = pd.DataFrame({
        "season": rows["season"], "date": rows["date"].dt.strftime("%Y-%m-%d"),
        "home_team": rows["home_team"], "away_team": rows["away_team"],
        "home_goals": rows["home_goals"].astype(int), "away_goals": rows["away_goals"].astype(int),
        "result": rows["result"], "elo_diff": rows["elo_diff"].round(1),
    })
    for name in ("model", "elo_only", "bookmakers"):
        games[name] = [[round(float(x), 4) for x in p] for p in r["probs"][name]]
    out = {
        "seasons": r["seasons"],
        "settings": {k: v for k, v in settings.items() if k != "tuning"},
        "overall": r["overall"], "per_season": r["per_season"],
        "calibration": {k: v.to_dict(orient="records") for k, v in r["calibration"].items()},
        "outcomes": OUTCOMES, "games": games.to_dict(orient="records"),
    }
    path.write_text(json.dumps(out, indent=1))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--validation", action="store_true",
                        help="report on the validation seasons instead of the test seasons")
    args = parser.parse_args()
    settings = load_settings()
    table = load_table()
    if args.validation:
        report(compare(before_test(table), VALIDATE, settings))
        return
    r = compare(table, TEST, settings)
    report(r)
    print(f"\nSaved {save(r, settings)}\n")
    print(markdown_table(r))


if __name__ == "__main__":
    main()
