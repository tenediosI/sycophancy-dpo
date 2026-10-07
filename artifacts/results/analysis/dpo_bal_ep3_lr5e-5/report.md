# Analysis: dpo_bal_ep3_lr5e-5

Each run is compared with `base` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; p_holm = Holm-corrected over all 21 tests; * = significant at 0.05 after correction. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 1299 | 90.6 | 27.1 | -63.5 [-66.2, -60.7] | 833 / 8 | 8.3e-235 | 1.7e-233 * |
| dpo_bal_ep3_lr5e-5_seed2 | 1297 | 90.7 | 63.8 | -26.8 [-29.6, -24.1] | 387 / 39 | 3.7e-73 | 6.3e-72 * |
| dpo_bal_ep3_lr5e-5_seed3 | 1296 | 90.6 | 55.5 | -35.1 [-37.9, -32.3] | 484 / 29 | 1.6e-107 | 3e-106 * |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 241 | 88.0 | 75.1 | -12.9 [-19.5, -6.2] | 49 / 18 | 0.00019 | 0.0029 * |
| dpo_bal_ep3_lr5e-5_seed2 | 240 | 87.9 | 85.4 | -2.5 [-7.9, 2.9] | 25 / 19 | 0.45 | 1 |
| dpo_bal_ep3_lr5e-5_seed3 | 241 | 88.0 | 85.9 | -2.1 [-7.9, 3.7] | 29 / 24 | 0.58 | 1 |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 1296 | 88.3 | 25.6 | -62.7 [-65.4, -59.9] | 825 / 13 | 1.6e-224 | 3.3e-223 * |
| dpo_bal_ep3_lr5e-5_seed2 | 1296 | 88.3 | 67.3 | -21.0 [-23.5, -18.4] | 310 / 38 | 3.5e-54 | 5.7e-53 * |
| dpo_bal_ep3_lr5e-5_seed3 | 1292 | 88.2 | 59.8 | -28.4 [-31.2, -25.7] | 405 / 38 | 1.3e-78 | 2.4e-77 * |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 238 | 71.8 | 70.2 | -1.7 [-8.0, 4.6] | 32 / 28 | 0.7 | 1 |
| dpo_bal_ep3_lr5e-5_seed2 | 237 | 72.2 | 71.7 | -0.4 [-6.3, 5.5] | 25 / 24 | 1 | 1 |
| dpo_bal_ep3_lr5e-5_seed3 | 238 | 71.8 | 77.7 | 5.9 [-0.4, 12.2] | 22 / 36 | 0.087 | 1 |

## mmlu

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 1140 | 60.4 | 60.5 | 0.2 [-0.9, 1.2] | 19 / 21 | 0.87 | 1 |
| dpo_bal_ep3_lr5e-5_seed2 | 1140 | 60.4 | 60.5 | 0.2 [-0.9, 1.1] | 16 / 18 | 0.86 | 1 |
| dpo_bal_ep3_lr5e-5_seed3 | 1140 | 60.4 | 60.4 | 0.1 [-0.9, 1.1] | 16 / 17 | 1 | 1 |

## gsm8k

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 500 | 57.6 | 55.6 | -2.0 [-5.0, 0.8] | 33 / 23 | 0.23 | 1 |
| dpo_bal_ep3_lr5e-5_seed2 | 500 | 57.6 | 55.4 | -2.2 [-5.2, 0.8] | 36 / 25 | 0.2 | 1 |
| dpo_bal_ep3_lr5e-5_seed3 | 500 | 57.6 | 56.0 | -1.6 [-4.6, 1.4] | 34 / 26 | 0.37 | 1 |

## ifeval

| run | n | base | run | diff [CI] | base-only / run-only | p | p_holm |
|---|---|---|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | 541 | 39.9 | 40.3 | 0.4 [-2.6, 3.3] | 32 / 34 | 0.9 | 1 |
| dpo_bal_ep3_lr5e-5_seed2 | 541 | 39.9 | 38.6 | -1.3 [-4.4, 1.8] | 40 / 33 | 0.48 | 1 |
| dpo_bal_ep3_lr5e-5_seed3 | 541 | 39.9 | 39.0 | -0.9 [-4.3, 2.2] | 41 / 36 | 0.65 | 1 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_bal_ep3_lr5e-5_seed1 | seen | -2.6 | 48.0 | 50.6 [43.6, 57.7] |
| dpo_bal_ep3_lr5e-5_seed1 | test_only | -16.4 | 44.6 | 61.0 [54.1, 67.9] |
| dpo_bal_ep3_lr5e-5_seed2 | seen | -2.8 | 21.6 | 24.3 [18.1, 30.3] |
| dpo_bal_ep3_lr5e-5_seed2 | test_only | -16.1 | 4.4 | 20.6 [14.2, 26.9] |
| dpo_bal_ep3_lr5e-5_seed3 | seen | -2.6 | 30.4 | 33.0 [26.5, 39.5] |
| dpo_bal_ep3_lr5e-5_seed3 | test_only | -16.4 | 17.9 | 34.3 [27.3, 41.1] |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 48.8 (19.3) | -41.8 (19.2) | -63.5 to -26.8 |
| acceptance/seen | 82.1 (6.1) | -5.8 (6.1) | -12.9 to -2.1 |
| capitulation/test_only | 50.9 (22.2) | -37.3 (22.2) | -62.7 to -21.0 |
| acceptance/test_only | 73.2 (4.0) | 1.3 (4.1) | -1.7 to 5.9 |
| mmlu | 60.5 (0.1) | 0.1 (0.1) | 0.1 to 0.2 |
| gsm8k | 55.7 (0.3) | -1.9 (0.3) | -2.2 to -1.6 |
| ifeval | 39.3 (0.9) | -0.6 (0.9) | -1.3 to 0.4 |
| discernment/seen | 33.3 (13.5) | 36.0 (13.4) | 24.3 to 50.6 |
| discernment/test_only | 22.3 (20.4) | 38.6 (20.5) | 20.6 to 61.0 |
