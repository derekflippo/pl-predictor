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
.plans/      Design spec
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
python backend/download_data.py   # downloads every season since 2000-01 (~27 files)
python backend/load_data.py       # combines them into backend/data/matches.csv
```

Run `download_data.py` again after each matchweek to get the latest
results. Past seasons are only downloaded once.

`backend/data/` is not committed to git. Anyone who clones the repo
rebuilds it with the two commands above.

### What's in `matches.csv`

One row per match, about 10,000 rows from August 2000 to today:

- `date`, `season`, `home_team`, `away_team`
- `home_goals`, `away_goals`, `result` (`H` / `D` / `A`), half-time goals
- shots, shots on target, corners, fouls, yellow and red cards
- `odds_home`, `odds_draw`, `odds_away`: pre-match betting odds (market
  average where available, otherwise a single bookmaker)
- `home_xg`, `away_xg`: expected goals, only from 2026-27

## Roadmap

0. [x] **Data pipeline**: download and clean every Premier League season
   since 2000-01
1. [ ] **More data**: Championship results (for promoted teams) and this
   season's fixtures from openfootball
2. [ ] **Features**: Elo ratings and recent form, with a test that no
   feature uses data from after the match
3. [ ] **Model**: logistic regression, evaluated season by season against
   "always pick home" (45.6%) and the bookmakers
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
