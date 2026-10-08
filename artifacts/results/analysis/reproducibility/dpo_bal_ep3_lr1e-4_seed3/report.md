# Analysis: reproducibility/dpo_bal_ep3_lr1e-4_seed3

Each run is compared with `dpo_bal_ep3_lr1e-4_seed3` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; p_holm = Holm-corrected over all 4 tests; * = significant at 0.05 after correction. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed3_rerun | 1300 | 19.2 | 20.5 | 1.3 [0.1, 2.5] | 26 / 43 | 0.053 | 0.21 |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed3_rerun | 241 | 75.1 | 75.1 | 0.0 [-2.5, 2.1] | 4 / 4 | 1 | 1 |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed3_rerun | 1300 | 19.5 | 19.8 | 0.4 [-0.8, 1.6] | 33 / 38 | 0.64 | 1 |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed3_rerun | 241 | 70.5 | 72.6 | 2.1 [-0.8, 5.4] | 5 / 10 | 0.3 | 0.91 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed3_rerun | seen | 55.9 | 54.6 | -1.3 [-4.0, 1.3] |
| dpo_bal_ep3_lr1e-4_seed3_rerun | test_only | 51.1 | 52.8 | 1.7 [-1.6, 5.1] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; p_holm over these tests.

| metric | n | base | runs | diff [CI] | p | p_holm |
|---|---|---|---|---|---|---|
| capitulation/seen | 1300 | 19.2 | 20.5 | 1.3 [0.1, 2.5] | 0.054 | 0.21 |
| acceptance/seen | 241 | 75.1 | 75.1 | 0.0 [-2.5, 2.1] | 1 | 1 |
| capitulation/test_only | 1300 | 19.5 | 19.8 | 0.4 [-0.8, 1.6] | 0.63 | 1 |
| acceptance/test_only | 241 | 70.5 | 72.6 | 2.1 [-0.8, 5.4] | 0.3 | 0.9 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 20.5 (-) | 1.3 (-) | 1.3 to 1.3 |
| acceptance/seen | 75.1 (-) | 0.0 (-) | 0.0 to 0.0 |
| capitulation/test_only | 19.8 (-) | 0.4 (-) | 0.4 to 0.4 |
| acceptance/test_only | 72.6 (-) | 2.1 (-) | 2.1 to 2.1 |
| discernment/seen | 54.6 (-) | -1.3 (-) | -1.3 to -1.3 |
| discernment/test_only | 52.8 (-) | 1.7 (-) | 1.7 to 1.7 |
