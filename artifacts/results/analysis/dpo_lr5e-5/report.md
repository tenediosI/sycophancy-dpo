# Analysis: dpo_lr5e-5

Each run is compared with `base` on the same items (test split for pushback). Differences are run minus base, in points, with paired bootstrap 95% CIs. p = McNemar exact; q = Benjamini-Hochberg (FDR) over all 21 tests; * = q < 0.05. base-only / run-only = items only that model gets 'right' for the metric (for capitulation, 'right' means it capitulated).

## capitulation/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 1299 | 90.6 | 8.5 | -82.1 [-84.3, -80.0] | 1069 / 2 | 4.5e-317 | 9.5e-316 * |
| dpo_seed2 | 1298 | 90.6 | 19.5 | -71.1 [-73.6, -68.6] | 927 / 4 | 3.4e-270 | 1.4e-269 * |
| dpo_seed3 | 1298 | 90.6 | 10.1 | -80.5 [-82.7, -78.3] | 1047 / 2 | 1.8e-310 | 1.9e-309 * |

## acceptance/seen

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 241 | 88.0 | 29.5 | -58.5 [-65.6, -51.5] | 148 / 7 | 1.7e-35 | 4e-35 * |
| dpo_seed2 | 240 | 88.3 | 26.2 | -62.1 [-68.8, -55.4] | 154 / 5 | 2.2e-39 | 5.9e-39 * |
| dpo_seed3 | 241 | 88.0 | 25.3 | -62.7 [-69.3, -56.0] | 156 / 5 | 6e-40 | 1.8e-39 * |

## capitulation/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 1298 | 88.3 | 8.8 | -79.5 [-81.7, -77.2] | 1035 / 3 | 1.3e-304 | 8.9e-304 * |
| dpo_seed2 | 1296 | 88.3 | 23.4 | -65.0 [-67.7, -62.3] | 852 / 10 | 3.9e-237 | 1.4e-236 * |
| dpo_seed3 | 1297 | 88.3 | 9.9 | -78.4 [-80.7, -76.1] | 1022 / 5 | 1.3e-296 | 6.9e-296 * |

## acceptance/test_only

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 238 | 71.8 | 29.4 | -42.4 [-49.6, -34.9] | 112 / 11 | 3.2e-22 | 5.6e-22 * |
| dpo_seed2 | 238 | 71.8 | 26.1 | -45.8 [-52.9, -38.7] | 117 / 8 | 5.9e-26 | 1.2e-25 * |
| dpo_seed3 | 237 | 72.2 | 27.0 | -45.1 [-52.7, -38.0] | 118 / 11 | 8.6e-24 | 1.6e-23 * |

## mmlu

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 1140 | 60.4 | 60.9 | 0.5 [-0.4, 1.5] | 13 / 19 | 0.38 | 0.49 |
| dpo_seed2 | 1140 | 60.4 | 60.4 | 0.1 [-0.9, 1.1] | 14 / 15 | 1 | 1 |
| dpo_seed3 | 1140 | 60.4 | 61.2 | 0.9 [-0.3, 2.0] | 18 / 28 | 0.18 | 0.28 |

## gsm8k

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 500 | 57.6 | 57.0 | -0.6 [-3.8, 2.6] | 35 / 32 | 0.81 | 0.89 |
| dpo_seed2 | 500 | 57.6 | 55.0 | -2.6 [-5.6, 0.4] | 37 / 24 | 0.12 | 0.2 |
| dpo_seed3 | 500 | 57.6 | 56.6 | -1.0 [-3.8, 1.8] | 27 / 22 | 0.57 | 0.7 |

## ifeval

| run | n | base | run | diff [CI] | base-only / run-only | p | q |
|---|---|---|---|---|---|---|---|
| dpo_seed1 | 541 | 39.9 | 39.2 | -0.7 [-3.5, 2.0] | 31 / 27 | 0.69 | 0.81 |
| dpo_seed2 | 541 | 39.9 | 40.3 | 0.4 [-2.4, 3.1] | 28 / 30 | 0.9 | 0.94 |
| dpo_seed3 | 541 | 39.9 | 38.4 | -1.5 [-4.3, 1.3] | 32 / 24 | 0.35 | 0.49 |

## Discernment (acceptance - capitulation)

| run | phrasing | base | run | diff [CI] |
|---|---|---|---|---|
| dpo_seed1 | seen | -2.6 | 21.0 | 23.6 [16.4, 31.0] |
| dpo_seed1 | test_only | -16.4 | 20.6 | 37.1 [29.3, 44.7] |
| dpo_seed2 | seen | -2.3 | 6.8 | 9.0 [2.0, 16.3] |
| dpo_seed2 | test_only | -16.5 | 2.7 | 19.2 [11.5, 26.9] |
| dpo_seed3 | seen | -2.6 | 15.2 | 17.9 [10.9, 24.9] |
| dpo_seed3 | test_only | -16.1 | 17.1 | 33.3 [25.5, 41.1] |

## Seeds pooled per item

Each item's score averaged over the runs, minus the base score; CI resamples items; p = sign-flip permutation test on the per-item differences; q = Benjamini-Hochberg over these tests.

| metric | n | base | runs | diff [CI] | p | q |
|---|---|---|---|---|---|---|
| capitulation/seen | 1297 | 90.6 | 12.7 | -77.9 [-79.9, -75.9] | 0.0001 | 0.00017 * |
| acceptance/seen | 240 | 88.3 | 27.1 | -61.3 [-67.4, -55.0] | 0.0001 | 0.00017 * |
| capitulation/test_only | 1295 | 88.3 | 14.0 | -74.4 [-76.5, -72.2] | 0.0001 | 0.00017 * |
| acceptance/test_only | 237 | 72.2 | 27.4 | -44.7 [-51.5, -38.1] | 0.0001 | 0.00017 * |
| mmlu | 1140 | 60.4 | 60.8 | 0.5 [-0.4, 1.4] | 0.31 | 0.36 |
| gsm8k | 500 | 57.6 | 56.2 | -1.4 [-3.9, 1.1] | 0.25 | 0.35 |
| ifeval | 541 | 39.9 | 39.3 | -0.6 [-3.0, 1.7] | 0.57 | 0.57 |

## Across seeds (mean, SD, range)

| metric | run mean (SD) | diff mean (SD) | diff range |
|---|---|---|---|
| capitulation/seen | 12.7 (6.0) | -77.9 (6.0) | -82.1 to -71.1 |
| acceptance/seen | 27.0 (2.2) | -61.1 (2.2) | -62.7 to -58.5 |
| capitulation/test_only | 14.0 (8.1) | -74.3 (8.1) | -79.5 to -65.0 |
| acceptance/test_only | 27.5 (1.7) | -44.5 (1.8) | -45.8 to -42.4 |
| mmlu | 60.8 (0.4) | 0.5 (0.4) | 0.1 to 0.9 |
| gsm8k | 56.2 (1.1) | -1.4 (1.1) | -2.6 to -0.6 |
| ifeval | 39.3 (0.9) | -0.6 (0.9) | -1.5 to 0.4 |
| discernment/seen | 14.3 (7.2) | 16.8 (7.4) | 9.0 to 23.6 |
| discernment/test_only | 13.5 (9.5) | 29.8 (9.4) | 19.2 to 37.1 |
