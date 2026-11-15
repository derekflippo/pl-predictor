# pl-predictor

Predicts Premier League results as three probabilities (home win, draw,
away win) and will show them on a static website. A portfolio and
learning project: the user is learning machine learning as we build.

## Where things stand (2026-11-15)

- **Done:** milestone 0 (data pipeline, PR #1) and milestone 1
  (Championship data, openfootball fixtures, team-name map, first tests,
  PR #2). Both are merged into `main`.
- **Next:** milestone 2 (Elo ratings and features) on branch
  `feature/features-elo`. Its spec is committed there. No code yet.
- **Specs:** read the right one before starting a milestone.
  - `.plans/pl-predictor-v1-spec.md`: the whole project, milestones 0–8
  - `.plans/milestone-2-features-spec.md`: milestone 2 in detail
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
pytest                            # 18 tests; data-dependent ones skip without downloads
```

- `backend/data/` is gitignored. Rebuild it with the commands above.
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
  - average shares every game: log loss 1.074
  - bookmakers: 54.5% accuracy, log loss 0.968
