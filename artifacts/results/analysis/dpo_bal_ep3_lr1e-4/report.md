# Analysis: dpo_bal_ep3_lr1e-4

Each run is compared with `base` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; p_holm = Holm-corrected over all 21 tests; * = significant at 0.05 after correction. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 1299 | 90.6 | 18.4 | -72.2 [-74.7, -69.7] | 945 / 7 | 7.3e-270 | 1.5e-268 * |
| dpo_bal_ep3_lr1e-4_seed2 | 1298 | 90.6 | 18.6 | -72.0 [-74.5, -69.4] | 944 / 10 | 2.2e-264 | 4.4e-263 * |
| dpo_bal_ep3_lr1e-4_seed3 | 1299 | 90.6 | 19.2 | -71.4 [-73.9, -68.7] | 939 / 12 | 1.1e-259 | 2e-258 * |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 241 | 88.0 | 71.4 | -16.6 [-23.2, -10.0] | 56 / 16 | 2.4e-06 | 3.6e-05 * |
| dpo_bal_ep3_lr1e-4_seed2 | 241 | 88.0 | 74.3 | -13.7 [-20.3, -7.1] | 52 / 19 | 0.00011 | 0.0016 * |
| dpo_bal_ep3_lr1e-4_seed3 | 241 | 88.0 | 75.1 | -12.9 [-19.5, -5.8] | 51 / 20 | 0.0003 | 0.0039 * |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 1298 | 88.3 | 16.4 | -71.9 [-74.4, -69.3] | 946 / 13 | 3.6e-260 | 6.8e-259 * |
| dpo_bal_ep3_lr1e-4_seed2 | 1298 | 88.3 | 18.3 | -70.0 [-72.7, -67.4] | 922 / 13 | 4.3e-253 | 7.3e-252 * |
| dpo_bal_ep3_lr1e-4_seed3 | 1298 | 88.3 | 19.4 | -68.9 [-71.5, -66.2] | 910 / 16 | 4.4e-245 | 7e-244 * |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 238 | 71.8 | 66.0 | -5.9 [-13.0, 1.3] | 47 / 33 | 0.15 | 1 |
| dpo_bal_ep3_lr1e-4_seed2 | 238 | 71.8 | 72.3 | 0.4 [-6.3, 7.1] | 34 / 35 | 1 | 1 |
| dpo_bal_ep3_lr1e-4_seed3 | 238 | 71.8 | 70.2 | -1.7 [-8.8, 5.5] | 41 / 37 | 0.73 | 1 |

## mmlu

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 1140 | 60.4 | 60.8 | 0.4 [-1.2, 2.1] | 43 / 48 | 0.68 | 1 |
| dpo_bal_ep3_lr1e-4_seed2 | 1140 | 60.4 | 60.1 | -0.3 [-1.8, 1.2] | 39 / 36 | 0.82 | 1 |
| dpo_bal_ep3_lr1e-4_seed3 | 1140 | 60.4 | 60.3 | -0.1 [-1.6, 1.5] | 41 / 40 | 1 | 1 |

## gsm8k

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 500 | 57.6 | 53.4 | -4.2 [-7.8, -0.6] | 56 / 35 | 0.035 | 0.43 |
| dpo_bal_ep3_lr1e-4_seed2 | 500 | 57.6 | 54.6 | -3.0 [-6.8, 0.8] | 53 / 38 | 0.14 | 1 |
| dpo_bal_ep3_lr1e-4_seed3 | 500 | 57.6 | 55.4 | -2.2 [-5.6, 1.4] | 46 / 35 | 0.27 | 1 |

## ifeval

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | 541 | 39.9 | 40.5 | 0.6 [-2.8, 3.9] | 40 / 43 | 0.83 | 1 |
| dpo_bal_ep3_lr1e-4_seed2 | 541 | 39.9 | 37.5 | -2.4 [-5.4, 0.6] | 42 / 29 | 0.15 | 1 |
| dpo_bal_ep3_lr1e-4_seed3 | 541 | 39.9 | 38.6 | -1.3 [-4.6, 1.8] | 43 / 36 | 0.5 | 1 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_bal_ep3_lr1e-4_seed1 | seen | -2.6 | 53.0 | 55.6 [48.6, 62.6] |
| dpo_bal_ep3_lr1e-4_seed1 | test_only | -16.4 | 49.6 | 66.0 [58.3, 73.7] |
| dpo_bal_ep3_lr1e-4_seed2 | seen | -2.6 | 55.6 | 58.3 [51.1, 65.4] |
| dpo_bal_ep3_lr1e-4_seed2 | test_only | -16.4 | 54.0 | 70.5 [63.0, 77.7] |
| dpo_bal_ep3_lr1e-4_seed3 | seen | -2.6 | 55.9 | 58.5 [51.2, 65.8] |
| dpo_bal_ep3_lr1e-4_seed3 | test_only | -16.4 | 50.8 | 67.2 [59.4, 74.9] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; p_holm over these tests.

| metric | n | base | runs | diff [CI] | p | p_holm |
|---|---|---|---|---|---|---|
| capitulation/seen | 1298 | 90.6 | 18.8 | -71.8 [-74.1, -69.5] | 0.0001 | 0.0007 * |
| acceptance/seen | 241 | 88.0 | 73.6 | -14.4 [-20.3, -8.3] | 0.0001 | 0.0007 * |
| capitulation/test_only | 1298 | 88.3 | 18.0 | -70.3 [-72.7, -67.8] | 0.0001 | 0.0007 * |
| acceptance/test_only | 238 | 71.8 | 69.5 | -2.4 [-8.8, 4.2] | 0.45 | 1 |
| mmlu | 1140 | 60.4 | 60.4 | 0.0 [-1.3, 1.4] | 1 | 1 |
| gsm8k | 500 | 57.6 | 54.5 | -3.1 [-6.1, -0.2] | 0.04 | 0.16 |
| ifeval | 541 | 39.9 | 38.9 | -1.0 [-3.7, 1.5] | 0.41 | 1 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 18.8 (0.4) | -71.8 (0.4) | -72.2 to -71.4 |
| acceptance/seen | 73.6 (2.0) | -14.4 (2.0) | -16.6 to -12.9 |
| capitulation/test_only | 18.0 (1.5) | -70.3 (1.5) | -71.9 to -68.9 |
| acceptance/test_only | 69.5 (3.2) | -2.4 (3.2) | -5.9 to 0.4 |
| mmlu | 60.4 (0.4) | 0.0 (0.4) | -0.3 to 0.4 |
| gsm8k | 54.5 (1.0) | -3.1 (1.0) | -4.2 to -2.2 |
| ifeval | 38.9 (1.5) | -1.0 (1.5) | -2.4 to 0.6 |
| discernment/seen | 54.8 (1.6) | 57.5 (1.6) | 55.6 to 58.5 |
| discernment/test_only | 51.4 (2.3) | 67.9 (2.3) | 66.0 to 70.5 |
