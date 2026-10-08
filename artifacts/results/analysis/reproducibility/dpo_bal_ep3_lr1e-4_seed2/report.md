# Analysis: reproducibility/dpo_bal_ep3_lr1e-4_seed2

Each run is compared with `dpo_bal_ep3_lr1e-4_seed2` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; q = Benjamini-Hochberg (FDR) over all 4 tests; * = q < 0.05. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed2_rerun | 1299 | 18.6 | 21.6 | 3.0 [1.7, 4.4] | 22 / 61 | 2.2e-05 | 8.7e-05 * |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed2_rerun | 241 | 74.3 | 78.4 | 4.1 [0.8, 7.5] | 3 / 13 | 0.021 | 0.043 * |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed2_rerun | 1300 | 18.3 | 19.7 | 1.4 [0.0, 2.8] | 32 / 50 | 0.06 | 0.08 |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed2_rerun | 241 | 71.4 | 71.4 | 0.0 [-2.9, 2.9] | 7 / 7 | 1 | 1 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed2_rerun | seen | 55.6 | 56.8 | 1.1 [-2.4, 4.7] |
| dpo_bal_ep3_lr1e-4_seed2_rerun | test_only | 53.1 | 51.7 | -1.4 [-4.7, 1.9] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; q = Benjamini-Hochberg over these tests.

| metric | n | base | runs | diff [CI] | p | q |
|---|---|---|---|---|---|---|
| capitulation/seen | 1299 | 18.6 | 21.6 | 3.0 [1.7, 4.4] | 0.0001 | 0.0004 * |
| acceptance/seen | 241 | 74.3 | 78.4 | 4.1 [0.8, 7.5] | 0.019 | 0.039 * |
| capitulation/test_only | 1300 | 18.3 | 19.7 | 1.4 [0.0, 2.8] | 0.059 | 0.078 |
| acceptance/test_only | 241 | 71.4 | 71.4 | 0.0 [-2.9, 2.9] | 1 | 1 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 21.6 (-) | 3.0 (-) | 3.0 to 3.0 |
| acceptance/seen | 78.4 (-) | 4.1 (-) | 4.1 to 4.1 |
| capitulation/test_only | 19.7 (-) | 1.4 (-) | 1.4 to 1.4 |
| acceptance/test_only | 71.4 (-) | 0.0 (-) | 0.0 to 0.0 |
| discernment/seen | 56.8 (-) | 1.1 (-) | 1.1 to 1.1 |
| discernment/test_only | 51.7 (-) | -1.4 (-) | -1.4 to -1.4 |
