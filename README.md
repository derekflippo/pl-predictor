# pl-predictor

Predicts Premier League match outcomes (home win, draw, away win) with a
machine learning model, and shows the predictions for upcoming games on a
web page.

The model uses football data only (results, form, team ratings). Betting
odds are not an input; they are the benchmark the model is measured
against. Predictions are locked before kickoff and scored afterwards, so
the site's track record is a real one.

## Project layout

```
backend/     Python: data download, features, model training, predictions
frontend/    React + Vite + Tailwind static site (coming in milestone 6)
.plans/      Design specs
docs/        Methodology explainer (source of a private page; reused in milestone 6)
```

The site is static: a daily GitHub Action retrains the model and writes the
predictions as JSON files, and the page reads them. There is no server.
See [.plans/pl-predictor-v1-spec.md](.plans/pl-predictor-v1-spec.md) for
the full design.

## Setup

Requires Python 3.10+.

```bash
pip install -r backend/requirements.txt
```

## Get the data

```bash
python backend/download_data.py   # Premier League + Championship since 2000-01, and this season's fixtures
python backend/load_data.py       # combines the results into backend/data/matches.csv
python backend/fixtures.py        # this season's fixtures matched to results, and the next matchweek
```

Run `download_data.py` again after each matchweek to get the latest
results. Past seasons are only downloaded once.

`backend/data/` is not committed to git. Anyone who clones the repo
rebuilds it with the commands above.

### What's in `matches.csv`

One row per match, about 24,500 rows from August 2000 to today: the
Premier League and the Championship. The Championship is there so that
promoted teams have a history.

- `league` (`E0` = Premier League, `E1` = Championship)
- `date`, `season`, `home_team`, `away_team`
- `home_goals`, `away_goals`, `result` (`H` / `D` / `A`), half-time goals
- shots, shots on target, corners, fouls, yellow and red cards
- `odds_home`, `odds_draw`, `odds_away`: pre-match betting odds (market
  average where available, otherwise a single bookmaker)
- `home_xg`, `away_xg`: expected goals, only from 2026-27

### Fixtures and team names

The schedule (dates, kickoff times, matchweeks) comes from openfootball.
Results always come from football-data. The two sources spell teams
differently ("Manchester United FC" vs "Man United"), so
[backend/team_names.csv](backend/team_names.csv) maps one to the other.
It covers every Premier League team since 2010-11. A newly promoted team
that isn't in the file stops the pipeline with an error that names it,
so it's never silently dropped.

Fixtures are matched to results by season and the two teams, not by date:
openfootball's dates can be out of date when games are moved.

## Model and results

The model is a softmax (multinomial) logistic regression on 29 features:
the Elo rating difference and level, Elo's expected score, and each team's
recent form (points, goals, shots on target, home or away points, rest
days, and whether it was just promoted). It uses an elastic net penalty
with C = 0.0316. Betting odds are not an input.

```bash
python backend/train.py --tune   # tunes on 2015-16 to 2018-19, writes backend/model_settings.json
python backend/evaluate.py       # the test: 2019-20 to 2025-26, season by season
python backend/train.py          # trains on every finished match and prints the weights
```

**How it was tested.** The settings were tuned on the validation seasons
(2015-16 to 2018-19) and committed before the test seasons were scored.
The test is walk-forward: each season is predicted by a model trained only
on the seasons before it. So every test game is one the system had never
seen.

Log loss on the test seasons (lower is better; 2,660 games):

| Season | Model | Elo only | Average shares | Bookmakers | Model accuracy | Bookmaker accuracy |
|---|---|---|---|---|---|---|
| 2019-20 | 0.974 | 0.979 | 1.066 | 0.973 | 51.8% | 53.2% |
| 2020-21 | 1.034 | 1.047 | 1.098 | 1.004 | 50.8% | 51.6% |
| 2021-22 | 0.957 | 0.958 | 1.074 | 0.936 | 55.8% | 57.9% |
| 2022-23 | 0.972 | 0.979 | 1.048 | 0.966 | 53.2% | 56.1% |
| 2023-24 | 0.925 | 0.933 | 1.057 | 0.909 | 57.9% | 58.9% |
| 2024-25 | 0.985 | 0.995 | 1.085 | 0.971 | 51.8% | 54.2% |
| 2025-26 | 1.036 | 1.036 | 1.082 | 1.015 | 47.6% | 49.5% |
| **All** | 0.983 | 0.990 | 1.073 | 0.968 | 52.7% | 54.5% |

- **Against Elo only** (logistic regression on the rating difference): the
  model is better by 0.0063 ± 0.0019 log loss. The form features add real
  information.
- **Against the bookmakers:** the model is behind by 0.0156 ± 0.0029. The
  bookmakers know about injuries, line-ups and news that the model can't see.
- **Always picking the home team** is right 43.4% of the time; the model's
  favourite is right 52.7% of the time.
- **2020-21**, played without crowds, is the model's worst season against
  the bookmakers. Home teams won 37.9% of games, and the model, which can't
  know whether fans are there, expected 46.1%.
- The model is well calibrated: when it says 60%, it happens about 60% of
  the time. The charts are in
  [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb).

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest
```

Some tests check the downloaded data and are skipped if you haven't run
`download_data.py` yet.

## Roadmap

0. [x] **Data pipeline**: download and clean every Premier League season
   since 2000-01
1. [x] **More data**: Championship results (for promoted teams) and this
   season's fixtures from openfootball
2. [x] **Features**: Elo ratings and recent form, with a test that no
   feature uses data from after the match
3. [x] **Model**: logistic regression, evaluated season by season against
   Elo only, average shares, "always pick home" and the bookmakers
4. [ ] **Predictions**: JSON for the site, and an append-only file of
   predictions locked before kickoff
5. [ ] **Daily GitHub Action**: retrain and lock predictions every day;
   the track record starts here
6. [ ] **Website (v1)**: upcoming games, track record, how it works;
   hosted on GitHub Pages
7. [ ] **v1.1**: team picker, key factors, head-to-head
8. [ ] **Later**: gradient boosting, a Poisson goals model

## Data sources

- Results, match stats and odds:
  [football-data.co.uk](https://www.football-data.co.uk/)
- Upcoming fixtures (planned):
  [openfootball](https://github.com/openfootball/football.json) (public domain)

Not betting advice.
