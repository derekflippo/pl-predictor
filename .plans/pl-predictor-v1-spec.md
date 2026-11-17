# PL Predictor: v1 spec

Written 2026-11-14, from the planning interview.

**Scope changed on 2026-11-16 (the user's decision).** There is no website,
no daily GitHub Action, no locked predictions or track record, and no
gradient boosting or Poisson model. The project ends with a local
`predict.py` that anyone can run after cloning the repo, and then the code
is published. Parts below that describe dropped work are marked
*Dropped*, and the rest of their text is removed. Git history has the
original.

## Summary

A portfolio project that predicts Premier League results (home win, draw,
away win) with a machine learning model trained on football data only.
Betting odds are not an input. They are the benchmark the model is measured
against. Anyone can clone the repo, download the data, retrain the model
and predict upcoming games on their own computer.

## Decisions

| Topic | Decision |
|---|---|
| Purpose | Portfolio and learning: clean code, a model you can explain, honest evaluation |
| Betting odds | Not a model input; used as the benchmark |
| Promoted teams | Also download Championship results so promoted teams get a real rating |
| Model | Multinomial logistic regression. No gradient boosting or Poisson model (dropped 2026-11-16) |
| Main metric | Log loss, with accuracy reported next to it |
| Draws in the output | Show all three probabilities and name the favorite; say "Close call" when the top two are within 5 points |
| Bookmaker comparison | In the evaluation only (README, notebook) |
| Track record | *Dropped 2026-11-16.* The walk-forward test on 2019-20 to 2025-26 is the evidence |
| Any match | `predict.py --home X --away Y` predicts any pairing of this season's teams |
| Architecture | Python scripts run on your own computer; no server and no website |
| Updates | Run by hand: download the latest results, then predict. The model retrains on every run |
| Publishing | The public GitHub repo: code, README results and the evaluation notebook |
| Tests | pytest + CI check on every pull request |
| Notebooks | Pipeline in plain scripts; one evaluation notebook for presentation |
| Git | One branch and pull request per milestone |
| v1 scope | Data, features, model, evaluation, and a local `predict.py` |

The original roadmap step "FastAPI endpoint" is dropped. So, since
2026-11-16, are the website, the daily Action, locking and the later models.

## Current state

Milestones 0 to 3 are built; see the README and the milestone specs. This
section described the state on 2026-11-14 and is no longer kept up to date.

## Architecture

```
On your own computer
  1. download_data.py   Premier League + Championship results (football-data),
                        this season's fixtures (openfootball)
  2. load_data.py       backend/data/matches.csv
  3. predict.py         builds features from earlier matches only, trains the
                        model on every finished match with the frozen
                        settings, and prints probabilities for upcoming games
```

- Raw downloads and everything in `backend/data/` stay out of git. Anyone
  who clones the repo rebuilds them with the commands above.
- `train.py --tune` and `evaluate.py` are for re-tuning and the evaluation.
  Predicting doesn't need them.

## Data

### Sources

| Data | Source | Use |
|---|---|---|
| Premier League results, stats, odds | football-data.co.uk `E0`, 2000-01 to now | Training, features, benchmark odds |
| Championship results | football-data.co.uk `E1`, 2000-01 to now | Ratings and form for promoted teams |
| This season's fixtures | openfootball `football.json`, `2026-27/en.1.json` | Upcoming games list, dates and kickoff times |

### Changes to the existing pipeline

- `download_data.py`: add the Championship (`E1`) and the openfootball
  fixtures file.
- `load_data.py`: add a `league` column (`E0` / `E1`).
- New `backend/team_names.csv` maps openfootball names to football-data
  names, for example `Manchester United FC` to `Man United` and
  `Nottingham Forest FC` to `Nott'm Forest`. The pipeline stops with an
  error when it finds an unmapped team, so a newly promoted club can't be
  silently dropped.

### Matching fixtures to results

Each home/away pair plays exactly once per season, so fixtures, locked
predictions and results are matched on `(season, home_team, away_team)`,
never on date. Postponed games then still match when they're played.

The openfootball results we checked were about two months behind
(through 2026-09-20 on 2026-11-14). We use openfootball only for the
schedule and take results from football-data.

## Features

Every feature for a match on date D is computed only from matches before
D. A test enforces this (see Testing).

| Feature | Notes |
|---|---|
| Elo rating, home and away, and the difference | Includes a home-advantage term. Ratings carry over between seasons with partial regression to the mean |
| Points per game, last 5 and last 10 | Windows continue across seasons |
| Goals for and against per game, last 5 and last 10 | |
| Shots on target for and against per game, last 5 and last 10 | Stands in for chance quality, because xG only exists from 2026-27 |
| Home team's home form and away team's away form, last 5 | |
| Days since each team's last league match | League games only; cup and European games are not in the data |
| Promoted-this-season flag | |

### Ratings across two leagues

Premier League and Championship teams never play each other in this data,
so the two leagues' Elo ratings are on different scales. When a team moves
up:

- Its starting Premier League rating = the average end-of-season rating of
  that summer's relegated teams, adjusted up or down by how far its
  Championship rating was above the Championship average.
- The size of that adjustment is fit on past seasons.
- A flat "default promoted rating" is kept as a comparison, to show the
  Championship data actually helps.

Relegated teams are handled the same way in reverse.

## Model

### Baselines (all reported)

1. Always pick the home team: 45.6% accuracy over all seasons.
2. Base rates: always predict the historical home / draw / away shares.
3. Elo only: logistic regression on the rating difference.
4. Bookmakers: implied probabilities from the pre-match average odds
   (1/odds, scaled to remove the bookmaker margin). This is the benchmark
   to beat.

### Model 1

Multinomial logistic regression (scikit-learn) on standardized features,
with regularization strength tuned by walk-forward validation.

### Later

*Dropped 2026-11-16:* no gradient boosting and no Poisson goals model.
Logistic regression is the final model.

### Evaluation

- Walk-forward by season: for each test season from 2019-20 to 2025-26,
  train on every season before it and test on it.
- Report log loss (main), accuracy and Brier score per season and on
  average, next to all baselines.
- Include a calibration plot: when the model says 60%, does it happen
  about 60% of the time?
- Expected range, to be confirmed by measurement: accuracy around 50–53%,
  with the bookmakers somewhat ahead on log loss.
- The results table goes in the README. The charts go in
  `notebooks/evaluation.ipynb`.

### Production model

Retrained on all finished matches each time `predict.py` runs, with the
settings frozen in `backend/model_settings.json`. Training takes seconds.
The settings are not re-tuned unless someone runs `train.py --tune` on
purpose.

## Predictions

`backend/predict.py` (milestone 4):

| Command | Predicts |
|---|---|
| `python backend/predict.py` | The next matchweek: the earliest matchweek with unplayed games from today on |
| `python backend/predict.py --matchweek 14` | The unplayed games of that matchweek |
| `python backend/predict.py --home Arsenal --away Chelsea` | Any pairing of this season's teams, as if played today (`--date` to change) |

- Each line shows the date, kickoff, teams, the three probabilities, and
  the favorite, or "Close call" when the top two are within 5 points.
- Team names are football-data's, as in `matches.csv` (`Man United`,
  `Nott'm Forest`). An unknown name stops with an error that lists this
  season's teams.
- Features are built in memory from `matches.csv` on every run, with the
  same code as the training rows (`features_for`), so they can't go stale.
- A game weeks away gets today's ratings and form. Results in between
  aren't simulated.
- Before a season's first result is in, `predict.py` stops with a message.
  Promoted and relegated teams' ratings are converted at the first game of
  the season, so earlier predictions would use the wrong scale.
- *Dropped 2026-11-16:* JSON for a site, `locked.jsonl`, locking rules and
  the in-season backtest.

## Website

*Dropped 2026-11-16.* The README and the evaluation notebook show the
results, and the code is the product.

## Repo layout

```
backend/
  download_data.py
  load_data.py
  fixtures.py
  team_names.csv
  elo.py
  features.py
  check_ratings.py
  train.py
  evaluate.py
  predict.py
  model_settings.json   frozen model settings (milestone 3)
  elo_settings.json
  tests/
  requirements.txt      pandas, scikit-learn
  requirements-dev.txt  pytest, jupyter, matplotlib
notebooks/
  evaluation.ipynb
docs/
  methodology.html      source of the private explainer page
.github/workflows/
  ci.yml                pytest on every pull request (milestone 5)
.plans/
```

## Testing

- **No leakage:** change the result of a future match and check that the
  features of every earlier match stay the same.
- **Team names:** every team in this season's openfootball file maps to a
  football-data name.
- **Predict:** probabilities are valid, unplayed games only, and unknown
  team names are rejected.
- **Probabilities:** each prediction's three probabilities are between 0
  and 1 and add up to 1.
- **Loader:** the 2003-04 and 2004-05 files load with 380 matches each.
- **CI:** `ci.yml` runs pytest on every pull request.

## Milestones

Each milestone is one branch and one pull request.

| # | Branch | Delivers |
|---|---|---|
| 0 | `setup/data-pipeline` | Commit and merge the existing pipeline; update the README roadmap to match this spec |
| 1 | `feature/championship-fixtures` | Championship data, openfootball fixtures, team name map, tests |
| 2 | `feature/features-elo` | Elo with promotion handling, feature table, leakage test |
| 3 | `feature/model-v1` | Baselines, logistic regression, walk-forward evaluation, notebook, README results |
| 4 | `feature/predict` | `predict.py`: retrain and predict the next matchweek or any named game, locally |
| 5 | `release/publish` | A README quick start that works from a fresh clone, `ci.yml` running pytest, data terms checked |

*Dropped 2026-11-16:* the old milestones 5 to 8 (daily Action and locking,
website v1, v1.1, gradient boosting and the Poisson model).

## Risks and open questions

- **Fixture dates:** openfootball lists upcoming matchweeks on the Saturday
  at 15:00 (and some midweek games at 20:00) until TV picks are made, and
  may be slow to update postponed games. `predict.py` shows the listed
  date, so a time can be a placeholder. Matching on team pairs avoids
  attaching the wrong result.
  - *To check:* football-data.co.uk may publish an upcoming-fixtures file
    that could cross-check dates.
- **Ratings across leagues:** the promotion mapping may not beat a flat
  default rating. Milestone 2 found the gain within noise.
- **Missing games:** cup and European matches aren't in the data, so rest
  days and fatigue are only partly known.
- **Data terms:** check football-data.co.uk's terms of use before
  milestone 5. The repo stores none of their files; users download them.

## Out of scope for v1

Injuries and lineups, live in-game updates, user accounts, other leagues,
club crests, anything aimed at betting, and (since 2026-11-16) a website,
a daily update, locked predictions, gradient boosting and a Poisson model.
