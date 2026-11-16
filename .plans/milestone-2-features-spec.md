# Milestone 2 spec: Elo ratings and features

Written 2026-11-14, from the milestone 2 interview. It builds on
[pl-predictor-v1-spec.md](pl-predictor-v1-spec.md) and doesn't repeat its
decisions.

Branch: `feature/features-elo`

## Goal

Turn the match history into one row of numbers per match that describes
both teams **before** kickoff. Milestone 3 trains the model on these rows.
The same code also produces rows for games not played yet (upcoming
fixtures and the team picker), so training and live predictions are
computed the same way.

## How the pieces fit

| Layer | What it is | Trained? |
|---|---|---|
| Elo (`elo.py`) | A fixed formula that updates each team's rating after every game, in date order | No. It has 4 settings: 3 tuned once, `N_home` fixed (see below) |
| Features (`features.py`) | Numbers describing both teams before a match; Elo is one of them | No |
| Model (milestone 3) | Logistic regression: features in, home/draw/away probabilities out | Yes, it learns one weight per feature |

## Decisions

| Topic | Decision |
|---|---|
| Win margin | Bigger wins move ratings more, with diminishing returns |
| Elo settings | Tuned on 2002-03 to 2018-19 only; 2019-20 on stays unseen for milestone 3 |
| Warm-up | Ratings start in August 2000; feature rows from 2002-03 on are used for training |
| Home advantage | Drifts with recent league-wide home results |
| Promoted teams' form | Championship games count, alongside opponent-adjusted form |
| Form measures | Raw and opponent-adjusted |
| Form windows | Fixed: last 5 and last 10 games |
| Future games | One code path for played and upcoming games |
| Championship rows | Built for both leagues with a `league` column; milestone 3 decides whether to train on them |
| Extra signals | None for now (no stage-of-season, Europe or manager features) |
| Teams new to the Championship | Start at the average rating of that summer's teams relegated from the Championship |
| Checks | All four checks below, written to allow for surprise seasons |

## Elo

### The formula

Each league has its own rating scale. Every team in both leagues starts
at 1500 in August 2000.

For each game:

1. **Expected score for the home team:**
   `E = 1 / (1 + 10 ^ (-(R_home + H - R_away) / 400))`
   - `R_home`, `R_away` are the two ratings before the game.
   - `H` is the current home advantage in rating points (see below).
   - `E` is between 0 and 1. The away team's expected score is `1 - E`.
2. **Actual score:** 1 for a home win, 0.5 for a draw, 0 for an away win.
3. **Win-margin multiplier `G`**, from the World Football Elo ratings:
   - draw or 1-goal win: 1
   - 2-goal win: 1.5
   - win by N >= 3 goals: (11 + N) / 8. A 3-0 is 1.75, and a 7-0 is 2.25.
4. **Update:** `R_home += K * G * (actual - E)`, and the away team changes
   by the same amount in the other direction.

### Home advantage that drifts

Before each game, `H` is worked out from that league's last `N_home`
games (all played before this date):

- `p` = the average actual home score over those games (win 1, draw 0.5)
- `H = 400 * log10(p / (1 - p))`

For example, `p = 0.58` (a normal season) gives `H` of about 56 points.
In the 2020-21 empty-stadium season, `p` fell to around 0.5, which gives
`H` near 0. `H` is also stored as a feature column, `home_advantage`.

### Season boundaries

At the start of each season, in this order:

1. **Teams changing league** keep their strength relative to their old
   league:
   - A promoted team's Premier League rating = the average end-of-season
     rating of the Premier League teams relegated that summer, plus
     `a * (its Championship rating - the Championship average)`.
   - A relegated team gets the same treatment in reverse.
   - A team new to the Championship (promoted from League One, which we
     don't download) starts at the average end-of-season rating of the
     teams relegated from the Championship that summer. A team coming back
     after years away is treated as new too, not given its stale rating.
2. **Summer shrink:** every rating moves part of the way to its league's
   average: `R = mean + (1 - s) * (R - mean)`.
3. **Re-center:** every team in a league is shifted by the same amount so
   the league average is 1500 again. Added on 2026-11-15: with `a > 0`,
   promoted teams arrive above the relegated teams they replace, so the
   Premier League average crept up every year (1504 to 1601 by 2026-27
   at `a = 0.25`, real numbers) and the Championship average fell. The
   shift changes no difference between teams in a league, so predictions
   and tuning are unaffected; it keeps the raw `elo` features comparable
   across seasons.

Who moved between leagues is found from which teams appear in each
league's season file. No hand-made list is needed.

### The four settings and how they're tuned

| Setting | Meaning | Values tried |
|---|---|---|
| `K` | How fast ratings move | 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40 |
| `s` | Summer shrink toward average | 0, 0.1, 0.2, 0.33, 0.5 |
| `N_home` | Games used for home advantage | Fixed at 380 (one season), not tuned |
| `a` | How much Championship strength carries over on promotion | 0, 0.25, 0.5, 0.75, 1 |

Changes agreed on 2026-11-15, after a rough tuning run:

- **K goes lower.** The rough best was K = 10, at the bottom of the
  first grid, so 5, 7.5 and 12.5 were added.
- **`N_home` is fixed at 380.** Home advantage changed little from
  2002-03 to 2018-19, so tuning would mostly fit luck and favour the
  slowest setting. The one big shift (empty stadiums, from June 2020) is
  in the unseen test seasons.

- **Score to minimize:** the mean squared error between Elo's expected
  score `E` and the actual score, over Premier League games from 2002-03
  to 2018-19. Games from 2019-20 onward are never used.
- **Grid:** 225 combinations (9 × 5 × 5). A pass over about 24,500
  games takes well under a second, so the full grid runs in a few
  minutes.
- **Choosing the winner:** the lowest MSE wins. Many combinations score
  within noise of each other, so the report also shows, for the runners-up:
  - the paired standard error of their difference from the best,
    computed per game
  - how many of the 17 seasons the best one scored lower in
- **Does Championship data help (`a > 0`)?** Compare the best `a > 0`
  combination with the best `a = 0` combination on Premier League games
  involving a promoted team, with a paired standard error. `a` only
  affects those games, so the full-grid score dilutes its effect.
- **`a = 0` is the flat default:** every promoted team gets the same
  rating. If the comparison doesn't show a clear gain for `a > 0`, the
  Championship data doesn't help ratings, and the milestone's write-up
  says so.
- **Saving:** `python backend/elo.py --tune` writes the chosen values and
  their score to `backend/elo_settings.json`, which is committed. The
  daily Action uses that file and never re-tunes.

## Features

One row per played match in either league. The columns:

**Identity and target:**
`league`, `season`, `date`, `home_team`, `away_team`, `result`,
`home_goals`, `away_goals`. The goals are kept for the later Poisson
model.

**Match-level:**
- `home_advantage`: `H` for this game
- `elo_expected`: Elo's expected home score `E`, including `H`

**Per team** (each feature below has a `home_` and an `away_` version):

| Feature | Definition |
|---|---|
| `elo` | Rating before the game |
| `ppg_5`, `ppg_10` | Points per game over the team's last 5 / 10 league games |
| `gf_5`, `gf_10`, `ga_5`, `ga_10` | Goals for / against per game, same windows |
| `sotf_5`, `sotf_10`, `sota_5`, `sota_10` | Shots on target for / against per game, same windows, using only games that have shot data |
| `vs_exp_5`, `vs_exp_10` | Results above expected: the average of (actual score - Elo expected score) over the window. Positive means better results than Elo predicted against those opponents |
| `venue_ppg_5` | Points per game over the last 5 home games (for the home team) or the last 5 away games (for the away team) |
| `rest_days` | Days since the team's last league game, capped at 21 so the summer break doesn't dominate |
| `came_up`, `came_down` | 1 if the team changed league since last season |
| `games_5`, `games_10` | How many games the window actually holds (fewer than 5/10 early on) |

Rules:

- **Windows** are a team's last N league games in either league, so they
  continue across seasons and across promotion. Opponent-adjusted form
  (`vs_exp`) corrects for weak Championship opponents.
- **Short windows** use the games available. A window with no games is
  left empty (NaN).
- **Missing values** stay as NaN in the table, and milestone 3 decides how
  to fill them. Expected cases:
  - the first weeks of 2000-01 (warm-up seasons, not used for training)
  - teams new to the Championship
  - 1 Championship match without shot data
- **The model only learns from these columns.** Model-time choices, such
  as using differences (`home_elo - away_elo`) or separate values, belong
  to milestone 3.

### No looking ahead

- Matches are processed **one date at a time**. All features for date D
  are computed before any result from date D is applied.
  - This matters for home advantage: another game on the same day must not
    leak into it.
- Every value for a match on date D comes only from matches before D.

## One code path for played and future games

- `features.py` keeps each team's running state (rating, recent games,
  last game date) and walks through the matches in date order.
- `build_features(matches)` returns the table for every played match.
- `features_for(games, matches)` returns rows for games that haven't
  happened (columns `league`, `date`, `home_team`, `away_team`), using the
  state after all played matches. It feeds:
  - upcoming fixtures in milestone 4
  - all 380 team pairs for the picker
- A future game weeks away uses today's state. Results in between are not
  simulated.
- **A test makes sure the two paths agree:** asking `features_for` about an
  already-played match, using only the matches before it, must give exactly
  the same row as `build_features`.

## Checks

Each check is explained below by what a failure would mean. They confirm
the code works. They don't measure how good the predictions are; that's
milestone 3's job.

### 1. End-of-season Elo vs the final table

- **What it compares:** for each finished Premier League season, the
  ranking by Elo **at the end of that season** against the final table.
  - The table is computed from the results: 3 points for a win, 1 for a
    draw, then goal difference, then goals scored.
- **Why a surprise season doesn't break it:**
  - The end-of-season rating already includes that season's results.
    A team that starts low and keeps winning climbs all season, so by May
    it ranks near the top. Leicester in 2015-16 is the classic case.
  - The check looks backward: both sides summarize the same games.
- **Why they won't match exactly:**
  - Elo weighs who you beat and by how much, and it carries over from
    earlier seasons.
  - The computed table ignores points deductions, which aren't in the
    data. For example, Portsmouth had 9 points deducted in 2009-10.
- **Measure:** the rank correlation per season (Spearman: 1 = identical
  order, 0 = unrelated).
  - The report lists the average and the 5 seasons with the biggest
    differences, with their teams, for a person to look at.
  - **Pass:** an average of at least 0.8 over 2002-03 to 2025-26. That's
    loose on purpose: it catches broken code without failing on surprise
    seasons.
- **For information only, never pass or fail:** the same comparison using
  ratings from the **start** of the season. That's a forecast, so it will
  score lower, and the gap shows how much seasons surprise. Teams like
  2015-16 Leicester will show up here, and that's expected.

### 2. Elo alone vs base rates

- **Setup:**
  - A logistic regression with one input, `elo_expected`, trained on
    Premier League games from 2002-03 to 2014-15.
  - Scored on 2015-16 to 2018-19, which stays inside the tuning window,
    so the milestone 3 test seasons stay unseen.
- **Pass:** lower log loss than predicting the training seasons' average
  home/draw/away shares for every game.
- **Failure means:** Elo carries no useful signal, which points to a bug.

### 3. Ratings chart

- In `notebooks/evaluation.ipynb`, the start of the milestone 3 notebook:
  ratings since 2000 for the clubs that spent the most seasons in the
  Premier League, with season boundaries marked.
- Sudden unexplained jumps or flat lines show bugs that numbers can hide.

### 4. Promoted teams (for information only)

- **Compare:** the average starting rating of promoted teams that went
  straight back down with the ones that stayed up. Report both averages,
  the counts and the standard error of the gap.
- **No pass or fail.** Changed on 2026-11-15. In our data, Championship
  strength barely predicts survival: the teams that went down had about
  the same Championship ratings as the ones that stayed up (+106 vs +102
  above the Championship average, real numbers), and tuning found no
  clear gain from `a > 0`. A pass mark would test luck. The unit tests
  check the conversion arithmetic instead.

## Files

```
backend/
  elo.py                 Elo formula, season boundaries, --tune
  elo_settings.json      tuned K, s, a and fixed N_home (committed)
  features.py            build_features, features_for; writes data/features.csv
  check_ratings.py       checks 1 and 2, check 4 for information; prints a report
  tests/test_elo.py
  tests/test_features.py
notebooks/
  evaluation.ipynb       check 3 (ratings chart)
```

Commands:

```
python backend/elo.py --tune      # once; writes elo_settings.json
python backend/features.py        # writes backend/data/features.csv and a summary
python backend/check_ratings.py   # the checks
```

`requirements.txt` gains `scikit-learn` (check 2). `requirements-dev.txt`
gains `jupyter` and `matplotlib` (check 3).

## Tests

- **Elo formula:** a hand-worked example. Equal ratings with `H = 0` give
  `E = 0.5`, and a home win moves each team by `K/2` in opposite
  directions. The multiplier gives 1, 1.5 and 1.75 for 1-, 2- and 3-goal
  wins.
- **Ratings balance:** within a league season, rating gains equal rating
  losses (the total stays constant until the season boundary).
- **Season boundary:** promoted and relegated teams are found correctly
  from a small made-up two-league dataset, and get the converted rating.
- **No leakage:** change the result of the last match in the data; every
  earlier row stays exactly the same.
- **Same day:** two games on the same date get the same `home_advantage`,
  and neither one's result affects the other's features.
- **Both paths agree:** `features_for` on a played match (with only
  earlier matches) equals that match's `build_features` row.
- **Windows:** a team with 3 past games has `games_5 = 3`, and its
  averages use those 3 games.
- **Checks 1, 2 and 4** run as tests when the downloaded data exists, and
  are skipped otherwise.

## Done when

- All tests pass.
- `elo_settings.json` is committed with its tuning score.
- `check_ratings.py` passes checks 1 and 2, reports check 4, and the
  notebook shows the chart.
- The pull request reports:
  - the tuned settings, including whether `a > 0`, which shows whether the
    Championship data helped
  - the check results, including the start-of-season comparison
  - how many NaN values each feature has

## Known limits

- Cup and European games aren't in the data, so `rest_days` misses
  midweek fixtures.
- The computed league table ignores points deductions.
- xG exists only from 2026-27, so it isn't a feature yet.
- A prediction for a game weeks away uses today's ratings and form.
