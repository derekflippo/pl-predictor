# pl-predictor

Predicts Premier League results as three probabilities: home win, draw and
away win. Clone the repo, download the data, and the model retrains and
predicts upcoming games on your own computer.

The model uses football data only: results, team ratings and recent form.
Betting odds are not an input; they are the benchmark it's measured
against. On 2,660 games it had never seen (2019-20 to 2025-26), its log
loss is 0.983, against 0.968 for the bookmakers and 0.990 for team ratings
alone. Its favourite won 52.7% of the time.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/derekflippo/pl-predictor.git
cd pl-predictor
pip install -r backend/requirements.txt
python backend/download_data.py   # 55 files from football-data.co.uk and openfootball
python backend/load_data.py       # combines them into backend/data/matches.csv
python backend/predict.py         # trains the model, then predicts the next matchweek
```

To keep the packages separate from the rest of your Python, make a
virtual environment before installing:

```bash
python -m venv .venv
source .venv/bin/activate
```

Tested with pandas 2.2 and 3.1, and scikit-learn 1.7 and 1.9. Both give
the same predictions.

## Predict upcoming games

```bash
python backend/predict.py                                 # the next matchweek
python backend/predict.py --matchweek 14                  # one matchweek's unplayed games
python backend/predict.py --home Arsenal --away Chelsea   # any pairing, as if played today
```

Each run builds the features from `backend/data/matches.csv`, trains the
model on every finished match (about 10 seconds), and prints the three
probabilities with the favourite, or "Close call" when the top two are
within 5 points:

```
Matchweek 11, 2026-27
  Sat 21 Nov 15:00          Chelsea v Leeds            home 52%  draw 27%  away 21%   Chelsea
  Sat 21 Nov 15:00        Newcastle v Arsenal          home 20%  draw 22%  away 58%   Arsenal
  Sat 21 Nov 15:00         Coventry v Crystal Palace   home 39%  draw 26%  away 35%   Close call
```

- Run `download_data.py` and `load_data.py` again after each matchweek, so
  ratings and form include the latest results. `predict.py` warns when
  games listed since the latest result have no result yet.
- Team names are football-data's: `Man United`, `Nott'm Forest`. A wrong
  name stops with a list of this season's teams.
- A game weeks away gets today's ratings and form.
- `--date` (for `--home`/`--away`) can't be earlier than the latest
  result, because the ratings and form already include every game up to
  then.
- Kickoff times are UK time. openfootball lists Saturday 15:00 until TV
  picks are made, so a listed time can be a placeholder.

### Or use the app

A local web page with the same predictions:

```bash
pip install -r backend/requirements-app.txt
streamlit run backend/app.py
```

It opens in your browser with three tabs:

- **Upcoming matchweek:** pick any matchweek; each game shows the three
  probabilities as bars and the favourite.
- **Any match:** pick a home team, an away team and a date. A "Why this
  prediction?" chart shows the 10 inputs that pushed it most toward a
  home or an away win.
- **Model:** how the home, draw and away probabilities change with the
  Elo gap (everything else at its average), and every input's weight.
The **Update data** button downloads the latest results, so you don't
need the command line at all. The model trains once when the page starts
(about 8 seconds); after that every prediction is instant.

## How it works

1. **Data.** Every Premier League and Championship result since 2000-01,
   with shots and pre-match odds, from football-data.co.uk. The
   Championship is there so promoted teams arrive with a history.
2. **Ratings and form.** The code walks through the matches in date
   order. For each game it records what was known before kickoff, then
   adds the result:
   - an Elo rating for every team, where bigger wins move ratings more and
     promoted teams are converted to the Premier League's scale
   - recent form over the last 5 and 10 games: points, goals, shots on
     target, home or away points, rest days, and whether the team was just
     promoted

   A test checks that no feature uses a result from on or after the match
   date.
3. **Model.** A softmax (multinomial) logistic regression on 29 of those
   numbers, with an elastic net penalty (C = 0.0316). It turns each game
   into three probabilities that add up to 100%.

### How it was tested

- **Tuning:** every choice (the penalty, its strength, which feature
  groups to keep) was made on the validation seasons, 2015-16 to 2018-19,
  by rules written before the search. The settings were then frozen in
  [backend/model_settings.json](backend/model_settings.json) and
  committed.
- **Test:** only after that were the test seasons, 2019-20 to 2025-26,
  scored, once. The test is walk-forward: each season is predicted by a
  model trained only on the seasons before it. So every test game is one
  the system had never seen.

## Results

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

- **Elo only** is logistic regression on the rating difference. The model
  beats it by 0.0063 ± 0.0019 log loss, so recent form adds real
  information.
- **Average shares** predicts the training seasons' home/draw/away shares
  for every game.
- **Bookmakers** are the pre-match odds with the margin removed. They're
  ahead of the model by 0.0156 ± 0.0029: they also know about injuries,
  line-ups and news.
- **Always picking the home team** is right 43.4% of the time.
- **2020-21**, played without crowds, is the model's worst season against
  the bookmakers. Home teams won 37.9% of games, and the model, which
  can't know whether fans are there, expected 46.1%.
- **Calibration:** when the model says 60%, it happens about 60% of the
  time. The charts are in
  [notebooks/evaluation.ipynb](notebooks/evaluation.ipynb).

To reproduce the results:

```bash
python backend/features.py       # builds backend/data/features.csv
python backend/evaluate.py       # the test seasons, season by season, next to every baseline
python backend/train.py          # trains on every finished match and prints the weights
python backend/train.py --tune   # re-runs the tuning on the validation seasons (rewrites model_settings.json)
```

## Project layout

```
backend/
  download_data.py   downloads the results and this season's fixtures
  load_data.py       combines the downloads into data/matches.csv
  fixtures.py        this season's fixtures, matched to results
  elo.py             Elo ratings across the two leagues
  features.py        one row of pre-match features per game
  train.py           the model, walk-forward training and tuning
  evaluate.py        scores the model and the baselines season by season
  metrics.py         log loss, Brier score, calibration
  predict.py         retrains and predicts upcoming games
  app.py             the same predictions as a local Streamlit page
  explain.py         the model's weights, its Elo curve, and why it predicted a match
  tests/
notebooks/           evaluation.ipynb: charts of the results
docs/                methodology.html: the method explained step by step
.plans/              design notes
```

`backend/data/` is not committed. The commands above rebuild it.

### The data in `matches.csv`

One row per match, about 24,500 rows from August 2000 to today:

- `league` (`E0` = Premier League, `E1` = Championship), `date`, `season`,
  `home_team`, `away_team`
- `home_goals`, `away_goals`, `result` (`H` / `D` / `A`), half-time goals
- shots, shots on target, corners, fouls, yellow and red cards
- `odds_home`, `odds_draw`, `odds_away`: pre-match odds (the market
  average where available, otherwise a single bookmaker)

The schedule comes from openfootball, which spells teams differently
("Manchester United FC" vs "Man United").
[backend/team_names.csv](backend/team_names.csv) maps one to the other,
and an unknown team stops the pipeline with an error that names it.
Fixtures are matched to results by season and the two teams, not by date,
so a moved game still finds its result.

## Tests

```bash
pip install -r backend/requirements-dev.txt
pytest
```

Tests that need the downloaded data skip if you haven't run
`download_data.py`. GitHub runs the tests on every pull request and every
push to `main`, on Python 3.10 and 3.12.

## Data sources

- Results, match stats and odds:
  [football-data.co.uk](https://www.football-data.co.uk/). Their
  [data page](https://www.football-data.co.uk/data.php) says the data is
  free, but intended for private individuals only, and not for commercial
  use. This repo doesn't include any of their files: `download_data.py`
  downloads them to your own computer, checks each one, and fetches past
  seasons only once. Please use the data on their terms.
- Fixtures:
  [openfootball](https://github.com/openfootball/football.json) (public domain)

Not betting advice.

## Licence

The code is under the [MIT licence](LICENSE). The licence doesn't cover
the data: football-data.co.uk's terms apply to their files.
