# pl-predictor

Predicts Premier League results as three probabilities (home win, draw,
away win). Anyone can clone the repo, retrain the model and predict
upcoming games locally with `predict.py`. There is no website (dropped on
2026-11-16). A portfolio and learning project: the user is learning
machine learning as we build.

## Status and key files

- **Current status and open threads:** in the session log's `STATUS.md`,
  which is printed at the start of each chat (see "Session log" below).
  On a machine without the log, use `git log`, the open pull requests and
  the v1 spec's milestone table.
- **Specs:** read the right one before starting a milestone.
  - `.plans/pl-predictor-v1-spec.md`: the whole project, milestones 0–5
    (the scope was cut on 2026-11-16: no website, no daily Action, no
    later models)
  - `.plans/milestone-2-features-spec.md`: milestone 2 in detail
  - `.plans/milestone-3-model-spec.md`: milestone 3, the model and its test
  - The decisions in them were made with the user in interviews. Don't
    change one without asking.
- **Methodology explainer:** a private page for the user, at
  https://claude.ai/artifact/X6HrP3fNfrDahR8yVA2Zh7. Its source is
  `docs/methodology.html`. Update it (republish to the same URL) when the
  method changes or real tuned values replace its examples.

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
pytest                            # data-dependent tests skip without downloads
```

- `backend/data/` is gitignored. Rebuild it with the commands above.
- CI (`.github/workflows/ci.yml`) runs pytest on Python 3.10 and 3.12 for
  every pull request. The data tests skip there.
- `train.make_model` supports scikit-learn before and after 1.8, which
  deprecated `penalty`. Both give the same model. Elastic net uses the
  saga solver with a fixed seed (`train.SEED`), so every run gives exactly
  the same model.
- Scores and the no-model baselines (log loss, Brier, paired SE,
  calibration, average shares, bookmakers) live in `backend/metrics.py`.
- Python is the user's Anaconda install, which already has pandas,
  scikit-learn and pytest.

## How we work

- One branch and one pull request per milestone. Branch names are in the
  v1 spec's milestone table. Merge only when the user says so.
- Commits are SSH-signed (`commit.gpgsign = true`). If a commit fails
  asking for a passphrase, the key isn't loaded. Ask the user to run
  `ssh-add --apple-use-keychain ~/.ssh/id_ed25519` in their own terminal.
  Never turn signing off.
- The `gh` CLI is signed in, so it can open and merge pull requests.
- Explain simply, with examples. The user often asks for plain-language
  explanations (`/anthropic-skills:ste`) and likes an interview before a
  milestone (`/anthropic-skills:interview`).
- Label numbers as real (computed from our data) or as examples.

## Session log

The log lives outside the repo, in `~/.claude/session-logs/pl-predictor/`,
because the repo is meant to be public (it was still private on
2026-11-16). A session-start hook in
`.claude/settings.local.json` (not committed) prints `STATUS.md` and the
last two daily logs into each new chat.

- **When to write:** after every finished piece of work, every decision,
  and every switch of task. Don't wait for the end of the session.
- **Where:** append an entry to today's file, `YYYY-MM-DD.md`, creating it
  if it's missing.
- **Entry format:** a `## HH:MM title` heading, then four `###` headings:
  - **What was done:** commands, files, features
  - **Decisions made:** and why
  - **Open threads:** unfinished work to pick up next time
  - **Errors:** what went wrong and how it was handled

  Write "None." under a heading with nothing to say.
- **Append only:** use `cat >> file <<'EOF'`. Never edit or delete
  earlier entries or older files. Other chats may be writing too.
- **Keep `STATUS.md` current:** rewrite it so it always shows where the
  project stands and every open thread, including threads from older
  days. Remove a thread only when it's done, and say so in that day's
  log.
- **Facts only,** in plain words. No drama, no praise.

## Data facts that matter

- football-data.co.uk files `E0` (Premier League) and `E1` (Championship).
  The 2003-04 and 2004-05 files have rows with extra empty fields at the
  end; `load_data.py` handles them.
- openfootball is used for the schedule only. Its results run about two
  months behind. It lists upcoming matchweeks on the Saturday at 15:00
  until TV picks are made.
  - Match fixtures to results on `(season, home_team, away_team)`, never
    on date.
- Tune Elo settings only on 2002-03 to 2018-19. The seasons from 2019-20
  on are the final test set for milestone 3.
- Benchmarks on the test seasons 2019-20 to 2025-26 (2,660 matches):
  - always picking home: 43.4% accuracy
  - average shares every game: log loss 1.073 (training shares, walk-forward)
  - bookmakers: 54.5% accuracy, log loss 0.968
  - Elo only: log loss 0.990
  - the model (milestone 3): 52.7% accuracy, log loss 0.983
- The test seasons were scored once, on 2026-11-16, with the settings
  frozen in commit 2f92dcd. Don't tune anything on them. A new model idea
  is judged on the validation seasons first.
