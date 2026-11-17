# pl-predictor

Predicts Premier League results as three probabilities (home win, draw,
away win) with a softmax logistic regression on Elo ratings and recent
form. Everything runs locally: download the data, then `predict.py`
retrains the model and predicts upcoming games. The README has the
results and how it was tested; `.plans/` has the design notes and the
decisions behind them.

## Commands

```bash
pip install -r backend/requirements-dev.txt
python backend/download_data.py   # results (E0 + E1 since 2000-01) and this season's fixtures
python backend/load_data.py       # builds backend/data/matches.csv
python backend/fixtures.py        # fixtures matched to results, next matchweek
python backend/elo.py --tune      # only when re-tuning; writes backend/elo_settings.json
python backend/features.py        # builds backend/data/features.csv
python backend/check_ratings.py   # rating checks 1, 2 and 4
python backend/train.py --tune    # only when re-tuning; writes backend/model_settings.json
python backend/train.py           # trains on every finished match, prints the weights
python backend/evaluate.py        # the test, 2019-20 to 2025-26; --validation for 2015-16 to 2018-19
python backend/predict.py         # retrains, predicts the next matchweek; --matchweek N; --home X --away Y
streamlit run backend/app.py      # the same as a local web page (pip install -r backend/requirements-app.txt)
pytest                            # data-dependent tests skip without downloads
```

- `backend/data/` is gitignored. Rebuild it with the commands above.
- CI (`.github/workflows/ci.yml`) runs pytest on Python 3.10 and 3.12 for
  every pull request. The data tests skip there.
- `train.make_model` supports scikit-learn before and after 1.8, which
  deprecated `penalty`. Both give the same model. Elastic net uses the
  saga solver with a fixed seed (`train.SEED`), so every run gives exactly
  the same model.
- `app.py` only calls `predict.py`'s functions and `download_data.download_all`;
  keep the logic in those modules so the CLI and the page always agree.
  Its tests use Streamlit's AppTest and skip without streamlit or data.
- Scores and the no-model baselines (log loss, Brier, paired SE,
  calibration, average shares, bookmakers) live in `backend/metrics.py`.

## Rules

- **The test seasons are spent.** 2019-20 to 2025-26 were scored once, on
  2026-11-16, with the settings frozen in commit 2f92dcd. Don't tune
  anything on them. A new model idea is judged on the validation seasons
  (2015-16 to 2018-19) first, by the rules in `train.TUNING_RULES`.
- Tune Elo settings only on 2002-03 to 2018-19.
- `elo.py --tune` and `train.py --tune` rewrite committed settings files.
  Run them only on purpose.
- A change that isn't meant to change the model must leave the results
  as they are: test log loss 0.9833, validation 0.95629, and the same
  `predict.py` output.

## Data facts that matter

- football-data.co.uk files `E0` (Premier League) and `E1` (Championship).
  The 2003-04 and 2004-05 files have rows with extra empty fields at the
  end; `load_data.py` handles them.
- football-data.co.uk's data is free for private individuals and not for
  commercial use. The repo never stores their files.
- openfootball is used for the schedule only. Its results run about two
  months behind. It lists upcoming matchweeks on the Saturday at 15:00
  until TV picks are made.
  - Match fixtures to results on `(season, home_team, away_team)`, never
    on date.
- Benchmarks on the test seasons 2019-20 to 2025-26 (2,660 matches):
  - always picking home: 43.4% accuracy
  - average shares every game: log loss 1.073 (training shares, walk-forward)
  - bookmakers: 54.5% accuracy, log loss 0.968
  - Elo only: log loss 0.990
  - the model: 52.7% accuracy, log loss 0.983
