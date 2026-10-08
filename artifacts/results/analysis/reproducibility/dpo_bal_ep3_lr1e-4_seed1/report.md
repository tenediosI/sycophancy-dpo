# Analysis: reproducibility/dpo_bal_ep3_lr1e-4_seed1

Each run is compared with `dpo_bal_ep3_lr1e-4_seed1` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; p_holm = Holm-corrected over all 4 tests; * = significant at 0.05 after correction. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1_rerun | 1300 | 18.4 | 19.2 | 0.8 [-0.5, 2.2] | 34 / 45 | 0.26 | 0.78 |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1_rerun | 241 | 71.4 | 71.4 | 0.0 [-3.3, 3.3] | 8 / 8 | 1 | 1 |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1_rerun | 1300 | 16.5 | 19.2 | 2.8 [1.5, 4.1] | 18 / 54 | 2.6e-05 | 0.0001 * |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1_rerun | 241 | 65.6 | 67.6 | 2.1 [-1.2, 5.8] | 7 / 12 | 0.36 | 0.78 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1_rerun | seen | 53.0 | 52.1 | -0.8 [-4.3, 2.7] |
| dpo_bal_ep3_lr1e-4_seed1_rerun | test_only | 49.1 | 48.4 | -0.7 [-4.4, 3.2] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; p_holm over these tests.

| metric | n | base | runs | diff [CI] | p | p_holm |
|---|---|---|---|---|---|---|
| capitulation/seen | 1300 | 18.4 | 19.2 | 0.8 [-0.5, 2.2] | 0.26 | 0.78 |
| acceptance/seen | 241 | 71.4 | 71.4 | 0.0 [-3.3, 3.3] | 1 | 1 |
| capitulation/test_only | 1300 | 16.5 | 19.2 | 2.8 [1.5, 4.1] | 0.0001 | 0.0004 * |
| acceptance/test_only | 241 | 65.6 | 67.6 | 2.1 [-1.2, 5.8] | 0.37 | 0.78 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 19.2 (-) | 0.8 (-) | 0.8 to 0.8 |
| acceptance/seen | 71.4 (-) | 0.0 (-) | 0.0 to 0.0 |
| capitulation/test_only | 19.2 (-) | 2.8 (-) | 2.8 to 2.8 |
| acceptance/test_only | 67.6 (-) | 2.1 (-) | 2.1 to 2.1 |
| discernment/seen | 52.1 (-) | -0.8 (-) | -0.8 to -0.8 |
| discernment/test_only | 48.4 (-) | -0.7 (-) | -0.7 to -0.7 |
