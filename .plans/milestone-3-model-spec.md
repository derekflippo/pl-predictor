# Milestone 3: the model and its test

Written 2026-11-16, at the end of the milestone. Milestone 3 had no
interview. This file records what was decided and why, and who decided
each point. The v1 spec's "Model" section is the brief this follows.

Branch: `feature/model-v1`. Code: `backend/train.py`, `backend/evaluate.py`,
`backend/model_settings.json`, `notebooks/evaluation.ipynb`.

## Decisions

| Topic | Decision | Made by |
|---|---|---|
| Model | Softmax logistic regression; median fill, then standardize; both fitted on training rows only | v1 spec |
| Splits | Train from 2002-03; validate 2015-16 to 2018-19; test 2019-20 to 2025-26 | Training plan agreed with the user |
| Walk-forward | Each season is predicted by a model trained on 2002-03 up to the season before. Used for both validation and test | v1 spec |
| Promoted teams' form | Raw form windows reset on a change of league (milestone 2 spec) | User, 2026-11-16 |
| games_5, games_10 | Not model inputs: a short window sits 8 to 11 SD from the mean and the model over-reacted | User, 2026-11-16 |
| came_down | Not a model input: always 0 in Premier League rows | Claude, first model |
| Elo inputs | `elo_diff` and `elo_sum` replace `home_elo` and `away_elo`: the same information, with weights that read as "gap" and "level". Same validation score at C = 1 (0.9585 both) | Claude, after the user asked why the difference isn't the main feature |
| Elo-only baseline | Logistic regression on `elo_diff`, as the v1 spec says. The first model compared with `elo_expected` instead | v1 spec |
| Tuning rules | Written in `train.tune` before the search ran. See below | Claude, following the agreed step 4 |
| Freeze | Settings committed in 2f92dcd before any test season was scored | Claude |

## Tuning (step 4), on the validation seasons

Rules:

- **C:** the one-standard-error rule over 17 values from 0.0001 to 1. The
  rule picks the smallest C (the strongest penalty) whose validation log
  loss is within one SE of the best C. The SE is the SE of the per-game
  difference from the best C.
- **Everything else:** the default is the L2 model on all 34 features,
  trained on Premier League rows only. A change is kept only when it is
  better by more than one SE.

Steps and real results (log loss on 1,520 validation games):

| Step | Result | Kept? |
|---|---|---|
| C for L2 | chosen 0.00316 (best 0.0178): 0.9591 | yes |
| L1 | 0.9578, −0.0014 ± 0.0016 vs L2 | no |
| Elastic net (l1_ratio 0.5) | 0.9570, −0.0022 ± 0.0010 vs L2, C 0.0316 | **yes** |
| Championship rows (with a 0/1 league flag) | 0.9562, −0.0008 ± 0.0015 | no |
| Ablations, one group at a time | removing `home_advantage` (−0.0001 ± 0.0001) or `vs_exp` (−0.0006 ± 0.0003) helped by more than one SE | both dropped |
| C again after the drops | 0.0316 | final: 0.9563 |

Ablation results worth knowing: removing shots on target costs the most
(+0.0037 ± 0.0016), then `came_up` (+0.0010 ± 0.0008) and `elo_sum`
(+0.0007 ± 0.0003). Removing goals (−0.0006 ± 0.0007) was within noise,
so goals stayed.

Final settings: elastic net, l1_ratio 0.5, C 0.0316, Premier League rows
only, 29 features. On validation the model is ahead of Elo only by
0.0040 ± 0.0022 and behind the bookmakers by 0.0116 ± 0.0036.

## The test: 2019-20 to 2025-26, scored once

Real results on 2,660 games:

| | Log loss | Accuracy | Brier |
|---|---|---|---|
| Model | 0.9833 | 52.7% | 0.5849 |
| Elo only | 0.9896 | 52.5% | 0.5887 |
| Average shares | 1.0729 | 43.4% | 0.6497 |
| Bookmakers | 0.9677 | 54.5% | 0.5745 |
| Always home | | 43.4% | |

- The model minus Elo only: −0.0063 ± 0.0019. Better in 6 of 7 seasons,
  with 2025-26 a tie.
- The model minus the bookmakers: +0.0156 ± 0.0029.
- This is inside the v1 spec's expected range: accuracy around 50–53%,
  with the bookmakers somewhat ahead.
- 2020-21, played without crowds: home teams won 37.9% of games, and the
  model expected 46.1%. It's the model's worst season against the
  bookmakers, +0.0299 ± 0.0101.
- Calibration: most groups are within 2 SE of the diagonal. Home wins at
  40–60% happen slightly less often than predicted, and away wins slightly
  more often, because the test seasons had fewer home wins than training.

## Code review fixes, 2026-11-17

A code review found that `tune()` dropped every feature group that passed
on its own, all at once, without checking the combined drop. The rule now
drops candidates one at a time, best first, each only if it's still better
by more than one SE than the features left (`TUNING_RULES` in `train.py`).
Re-running the fixed `tune()` reproduces the frozen settings exactly:
`vs_exp` passes first, then `home_advantage`, and the final validation log
loss is the same 0.95629. `model_settings.json` was not rewritten, so its
`rule` text is the original wording.

The same review found that saga, the elastic net solver, shuffled rows
randomly, so runs differed by up to 0.000006 in probability (0.00000003 in
test log loss). It now uses a fixed seed. The reported numbers are
unchanged.

## Production model

`train.train_final` trains on every finished Premier League match from
2002-03 with the frozen settings. Milestone 4's `predict.py` calls it
daily. The weights it prints are the ones the site will use.

## Rules from now on

- Never tune on the test seasons. A new idea (a feature, a model) is
  judged on the validation seasons first, by the same rules, and then
  tested once.
- Re-running `train.py --tune` overwrites `model_settings.json`. Do it only
  on purpose, and record why in the session log.

## Open questions

- **Home advantage drift.** Home wins fell in the test seasons. The
  `home_advantage` feature tracks this and was dropped on validation by a
  margin of 0.0001. That's a question for the next model change, judged
  on validation first.
- **Draws in lopsided games.** On validation, games where the model gave a
  draw 10–20% ended in a draw 11.8% of the time (−2.3 SE). On the test
  this group was fine (17.2% predicted, 16.9% happened).
- **Championship rows** scored slightly better (−0.0008 ± 0.0015) but not
  by the margin the rule asks for. Worth another look if more seasons
  arrive.
- Gradient boosting and a Poisson goals model come in milestone 8, judged
  the same way.
