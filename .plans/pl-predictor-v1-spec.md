# PL Predictor: v1 spec

Written 2026-11-14, from the planning interview.

## Summary

A portfolio project that predicts Premier League results (home win, draw,
away win) with a machine learning model trained on football data only.
Betting odds are not an input. They are the benchmark the model is measured
against. A static website shows upcoming fixtures with the model's
probabilities and a public track record of predictions that were locked
before kickoff.

## Decisions

| Topic | Decision |
|---|---|
| Purpose | Portfolio and learning: clean code, a model you can explain, honest evaluation |
| Betting odds | Not a model input; used as the benchmark |
| Promoted teams | Also download Championship results so promoted teams get a real rating |
| Model | Classifier first (multinomial logistic regression); Poisson goals model later as a comparison |
| Main metric | Log loss, with accuracy reported next to it |
| Draws in the UI | Show all three probabilities and highlight the favorite; label "Close call" when the top two are within 5 points |
| Match card (v1) | Probability bar plus recent form strip |
| Match card (v1.1) | Key factors, head-to-head |
| Bookmaker comparison | On the Track record page only, not on match cards |
| Track record | Lock predictions before kickoff and score them after; earlier games this season shown separately as a labeled backtest |
| Team picker | Yes, in v1.1: all 380 home/away pairs precomputed |
| Architecture | Python precomputes JSON; static React site reads it; no live API server |
| Updates | GitHub Action runs daily: download, retrain, lock, publish |
| Frontend | React + Vite + Tailwind |
| Hosting | GitHub Pages (`derekflippo.github.io/pl-predictor`) |
| Look | Clean, light and dark mode, club-color stripe per team, no crests (trademarks) |
| Tests | pytest + CI check on every pull request |
| Notebooks | Pipeline in plain scripts; one evaluation notebook for presentation |
| Git | One branch and pull request per milestone |
| v1 scope | Model, upcoming fixtures page, track record. Locking starts as early as possible |

The original roadmap step "FastAPI endpoint" is dropped. Precomputed JSON
covers every view in this spec, including the team picker.

## Current state

- `backend/download_data.py` downloads Premier League CSVs (football-data
  code `E0`) for 2000-01 to the current season.
- `backend/load_data.py` combines them into `backend/data/matches.csv`:
  9,980 matches, 2000-08-19 to 2026-11-08, no missing odds or shot stats.
- These files are on branch `setup/data-pipeline`, staged but **not yet
  committed**: commit signing needs the SSH key loaded
  (`ssh-add --apple-use-keychain ~/.ssh/id_ed25519`).

## Architecture

```
GitHub Action (daily, 06:00 UTC)
  1. download_data.py   Premier League + Championship results (football-data),
                        this season's fixtures (openfootball)
  2. features.py        one feature row per match, from earlier matches only
  3. train.py           fit the model on all finished matches
  4. predict.py         write JSON for the site; append newly locked
                        predictions to backend/predictions/locked.jsonl
  5. commit locked.jsonl and the JSON, then build and deploy the site

React site (static, GitHub Pages)
  reads frontend/public/data/*.json; no server
```

- Raw downloads stay out of git. The Action caches `backend/data/raw/` with
  `actions/cache`, so past seasons are not downloaded again every day.
- `backend/predictions/locked.jsonl` is committed. It is the permanent
  record and must never be rewritten.
- The Action commits as the GitHub Actions bot, so the local SSH signing
  setup does not affect it. Its data commits go straight to `main`. Code
  changes still go through branches and pull requests.

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

Gradient boosting (XGBoost or LightGBM), then a Poisson goals model that
also gives likely scorelines. Each is compared against Model 1 with the
same evaluation.

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

Retrained daily on all finished matches. Training on about 10,000 rows
takes seconds. Each locked prediction records the commit SHA of the code
that made it.

## Predictions and locking

`predict.py` writes:

| File | Contents |
|---|---|
| `fixtures.json` | Upcoming games in the next 14 days, grouped by matchweek: teams, kickoff time, three probabilities, favorite or "Close call", each team's last 5 results |
| `record.json` | Every scored locked prediction, plus summary numbers for the model and the bookmakers, and the backtest section |
| `pairs.json` (v1.1) | Probabilities for all 380 home/away pairs of this season's teams |
| `meta.json` | Last update time, data range, model version |

### Locking rules

- Each daily run locks every fixture that kicks off within the next 48
  hours and isn't locked yet.
- A game therefore gets locked 24–48 hours before kickoff. If one
  scheduled run is delayed or skipped, the next run still locks it in
  time.
- A locked line contains: season, matchweek, date, kickoff, home, away,
  p_home, p_draw, p_away, model_version, locked_at.
- `locked.jsonl` is append-only. Existing lines are never changed. A test
  enforces this.
- When a result arrives in football-data, the prediction is scored. The
  bookmaker comparison uses the pre-match odds from that same results
  row, the same odds the backtest uses.
- A postponed game keeps its original lock and is scored whenever it's
  played.

### Backtest for this season

Games this season played before locking starts get a backtest. For each
matchweek, the model is trained on data from before that matchweek and
predicts its games. The site shows these in a separate section labeled
"Backtest" and never mixes them into the locked record.

## Website

### Pages (v1)

1. **Upcoming** (home page): games grouped by matchweek. Each card shows:
   - both teams with a club-color stripe
   - kickoff in the viewer's local time
   - a three-part probability bar, with the favorite highlighted, or a
     "Close call" label
   - each team's last 5 results as W/D/L badges
2. **Track record**:
   - summary tiles: games scored, accuracy and log loss for the model and
     the bookmakers
   - a table of locked predictions with results
   - the backtest section, below and clearly labeled
3. **How it works**: the model in plain words, the features, data sources
   with credit to football-data.co.uk and openfootball, and a "not betting
   advice" note.

### v1.1

- **Team picker**: choose a home team and an away team and see the
  prediction, read from `pairs.json`.
- **Key factors** on each card, for example "Arsenal: 13 pts from last 5,
  rating 1820 vs 1610", from the largest feature contributions in the
  logistic regression.
- **Head-to-head**: recent results between the two teams.

### Design

- Tailwind, mobile first.
- Light and dark mode.
- Neutral colors, with one primary color per club in
  `frontend/src/data/clubs.json`.
- No crests or official logos.
- Vite base path `/pl-predictor/` for GitHub Pages.

## Repo layout (target)

```
backend/
  download_data.py      done; add E1 + fixtures
  load_data.py          done; add league column
  team_names.csv
  elo.py
  features.py
  train.py
  evaluate.py
  predict.py
  predictions/locked.jsonl
  tests/
  requirements.txt      pandas, scikit-learn
  requirements-dev.txt  pytest, jupyter, matplotlib
notebooks/
  evaluation.ipynb
frontend/               Vite + React + Tailwind
.github/workflows/
  ci.yml                pytest + frontend build on every PR
  daily.yml             daily pipeline, commit, deploy
.plans/
```

## Testing

- **No leakage:** change the result of a future match and check that the
  features of every earlier match stay the same.
- **Team names:** every team in this season's openfootball file maps to a
  football-data name.
- **Locking:** running `predict.py` twice never changes existing lines in
  `locked.jsonl`, and it never locks a game that has already kicked off.
- **Probabilities:** each prediction's three probabilities are between 0
  and 1 and add up to 1.
- **Loader:** the 2003-04 and 2004-05 files load with 380 matches each.
- **CI:** `ci.yml` runs pytest and `npm run build` on every pull request.

## Milestones

Each milestone is one branch and one pull request. Locking starts before
the website exists, so the track record grows from as early as possible.

| # | Branch | Delivers |
|---|---|---|
| 0 | `setup/data-pipeline` | Commit and merge the existing pipeline; update the README roadmap to match this spec |
| 1 | `feature/championship-fixtures` | Championship data, openfootball fixtures, team name map, tests |
| 2 | `feature/features-elo` | Elo with promotion handling, feature table, leakage test |
| 3 | `feature/model-v1` | Baselines, logistic regression, walk-forward evaluation, notebook, README results |
| 4 | `feature/predict-lock` | `predict.py`, JSON outputs, `locked.jsonl`, backtest |
| 5 | `feature/daily-action` | `ci.yml` and `daily.yml` (locking only, no deploy yet). **The track record starts here** |
| 6 | `feature/frontend-v1` | Upcoming, Track record and How it works pages; deploy to GitHub Pages. **v1 shipped** |
| 7 | `feature/v1.1` | Team picker, key factors, head-to-head |
| 8 | later | Gradient boosting, Poisson goals model, model comparison page |

## Risks and open questions

- **Fixture dates:** openfootball may be slow to update postponed games.
  Matching on team pairs avoids wrong scores. A postponed game may be
  locked days before it is actually played.
  - *To check:* football-data.co.uk may publish an upcoming-fixtures file
    that could cross-check dates.
- **Scheduled Action reliability:** GitHub can delay or skip scheduled
  runs. The 48-hour lock window covers one missed run.
- **Scheduled workflows get disabled:** GitHub disables them in public
  repos after 60 days without repository activity.
  - *To check:* whether the daily bot commits count as activity.
- **Ratings across leagues:** the promotion mapping may not beat a flat
  default rating. The evaluation decides which one stays.
- **Missing games:** cup and European matches aren't in the data, so rest
  days and fatigue are only partly known.
- **Commit signing:** local commits need the SSH key in the agent. If
  branch protection on `main` later requires signed commits, the bot's
  data commits need an exception.
- **Data terms:** football-data.co.uk is free to use with credit. The repo
  stores only our own predictions, never their files.

## Out of scope for v1

Injuries and lineups, live in-game updates, user accounts, other leagues,
club crests, and anything aimed at betting.
