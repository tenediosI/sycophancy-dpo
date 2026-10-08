# Analysis: reproducibility/dpo_seed1

Each run is compared with `dpo_seed1` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; q = Benjamini-Hochberg (FDR) over all 4 tests; * = q < 0.05. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1_rerun | 1300 | 8.5 | 9.6 | 1.2 [0.2, 2.2] | 16 / 31 | 0.04 | 0.16 |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1_rerun | 241 | 29.5 | 29.5 | 0.0 [-2.9, 2.9] | 7 / 7 | 1 | 1 |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1_rerun | 1300 | 8.8 | 9.4 | 0.5 [-0.5, 1.5] | 18 / 25 | 0.36 | 0.72 |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1_rerun | 241 | 29.5 | 30.3 | 0.8 [-2.1, 3.7] | 5 / 7 | 0.77 | 1 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_seed1_rerun | seen | 21.0 | 19.8 | -1.2 [-4.4, 2.1] |
| dpo_seed1_rerun | test_only | 20.6 | 20.9 | 0.3 [-2.7, 3.2] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; q = Benjamini-Hochberg over these tests.

| metric | n | base | runs | diff [CI] | p | q |
|---|---|---|---|---|---|---|
| capitulation/seen | 1300 | 8.5 | 9.6 | 1.2 [0.2, 2.2] | 0.037 | 0.15 |
| acceptance/seen | 241 | 29.5 | 29.5 | 0.0 [-2.9, 2.9] | 1 | 1 |
| capitulation/test_only | 1300 | 8.8 | 9.4 | 0.5 [-0.5, 1.5] | 0.36 | 0.73 |
| acceptance/test_only | 241 | 29.5 | 30.3 | 0.8 [-2.1, 3.7] | 0.78 | 1 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 9.6 (-) | 1.2 (-) | 1.2 to 1.2 |
| acceptance/seen | 29.5 (-) | 0.0 (-) | 0.0 to 0.0 |
| capitulation/test_only | 9.4 (-) | 0.5 (-) | 0.5 to 0.5 |
| acceptance/test_only | 30.3 (-) | 0.8 (-) | 0.8 to 0.8 |
| discernment/seen | 19.8 (-) | -1.2 (-) | -1.2 to -1.2 |
| discernment/test_only | 20.9 (-) | 0.3 (-) | 0.3 to 0.3 |
