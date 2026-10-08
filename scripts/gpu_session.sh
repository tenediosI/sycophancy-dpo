#!/usr/bin/env bash
# Full GPU session on a Vast.ai instance, then push results and stop:
#
#   bash scripts/gpu_session.sh 2>&1 | tee -a session.log
#
#   1. (GENERATE=1 only) Regenerate preference pairs (train, val).
#   2. Base model references on val, test and capability (skipped when they exist).
#   3. DPO learning-rate sweep, seed 0, each evaluated on val.
#   4. Pick the learning rate with scripts/pick_lr.py (rule fixed in advance, val only).
#      If no rate qualifies, stop here. With LR set, steps 3-4 are skipped and steps 5-6
#      run for each given rate instead.
#   5. DPO with that rate for seeds 1-3, each evaluated on test (pushback + capability).
#   6. Step 7 statistics against the base model (stage=analyse).
#
# The defaults are GPU session 4: balanced pairs (1 wrong-pushback pair per
# correct-pushback pair), 3 epochs and the constrained rule. Earlier sessions:
#   session 2: GENERATE=1 RUN=dpo RATIO=3 RULE=gap EPOCHS=1 LRS="5e-6 2e-5 5e-5"
#   session 3: RUN=dpo_bal RATIO=1 RULE=constrained EPOCHS=1 LRS="2e-5 3e-5 4e-5 5e-5"
#              (no rate qualified: 24 steps per run barely moved the model)
#   session 4: the defaults (no rate qualified: acceptance fell 8 points at 5e-5)
#   session 5: LR="5e-5 1e-4" with the defaults (two points on the trade-off curve,
#              chosen by hand from session 4's val sweep, i.e. post hoc)
#
# Results are committed and pushed after every phase, and once more on exit. The instance
# is stopped on exit whether the session succeeds or fails. Rerunning the script skips
# finished steps and redoes an interrupted one.
#
# Environment overrides:
#   RUN="dpo_bal_ep3"        run-name stem: <RUN>_lr<lr>_seed0 (sweep), <RUN>_seed<s> (final)
#   RATIO=1                  train.pairs.wrong_per_correct
#   EPOCHS=3                 train.num_train_epochs
#   RULE=constrained         pick_lr.py rule: constrained | gap
#   LRS="2e-5 3e-5 5e-5 1e-4"  learning rates to sweep
#   SEEDS="1 2 3"            final training seeds
#   LR="5e-5 1e-4"           skip the sweep; final runs <RUN>_lr<lr>_seed<s> for each rate
#   GENERATE=1               regenerate the preference pairs first
#   PREFIX=""                prefix for run names, e.g. "debug/" (for local tests)
#   EXTRA=""                 extra Hydra overrides for every run.py call, e.g. "experiment=debug"
#   NO_PUSH=1                do not commit or push
#   (the instance is only stopped when $CONTAINER_ID is set, i.e. on Vast)
set -euo pipefail

RUN=${RUN:-"dpo_bal_ep3"}
RATIO=${RATIO:-1}
EPOCHS=${EPOCHS:-3}
RULE=${RULE:-"constrained"}
LRS=${LRS:-"2e-5 3e-5 5e-5 1e-4"}
SEEDS=${SEEDS:-"1 2 3"}
PREFIX=${PREFIX:-""}
EXTRA=${EXTRA:-""}
BASE="${PREFIX}base"

source "$(dirname "$0")/lib.sh"

say "Session: RUN=$RUN RATIO=$RATIO EPOCHS=$EPOCHS RULE=$RULE LRS=\"$LRS\" SEEDS=\"$SEEDS\""

# 1. Preference pairs (overwrite: replaces the committed ones).
if [ -n "${GENERATE:-}" ]; then
    for split in train val; do
        run stage=generate generate.split="$split" overwrite=true
    done
    push "preference pairs"
fi

# 2. Base model references (skipped when they already exist).
[ -f "artifacts/results/$BASE/val/summary.json" ] || run stage=evaluate eval.split=val eval.run_name="$BASE"
[ -f "artifacts/results/$BASE/test/summary.json" ] || run stage=evaluate eval.split=test eval.run_name="$BASE"
[ -f "artifacts/results/$BASE/capability/summary.json" ] || run stage=capability capability.run_name="$BASE"
push "base model references"

# Steps 5-6 for one learning rate: fresh seeds evaluated on test, then statistics
# against the base model.
final_runs() {  # lr stem
    local lr=$1 stem=$2 finals=() name
    for seed in $SEEDS; do
        name="${PREFIX}${stem}_seed${seed}"
        train_dpo "$name" "$lr" "$seed" train.pairs.wrong_per_correct="$RATIO" train.num_train_epochs="$EPOCHS"
        evaluate "$name" test
        capability "$name"
        finals+=("$name")
        push "$name (lr $lr) on test"
    done
    local runs_list
    runs_list=$(IFS=,; echo "${finals[*]}")
    run stage=analyse analyse.name="${PREFIX}${stem}" analyse.base_run="$BASE" "analyse.runs=[$runs_list]" overwrite=true
    push "$stem analysis"
}

if [ -n "${LR:-}" ]; then
    # Learning rates chosen by hand (after a sweep): no sweep, final runs for each.
    say "LR=\"$LR\" given: skipping the sweep and the selection rule."
    for lr in $LR; do
        final_runs "$lr" "${RUN}_lr${lr}"
    done
    say "All done."
    exit 0
fi

# 3. Learning-rate sweep on val, seed 0.
sweep=()
for lr in $LRS; do
    name="${PREFIX}${RUN}_lr${lr}_seed0"
    train_dpo "$name" "$lr" 0 train.pairs.wrong_per_correct="$RATIO" train.num_train_epochs="$EPOCHS"
    evaluate "$name" val
    sweep+=("$lr=$name")
done

# 4. Choose the learning rate (last line of the output).
best_lr=$(python scripts/pick_lr.py --rule "$RULE" --out "$RUN" "$BASE" "${sweep[@]}" | tee /dev/stderr | tail -n 1)
say "Chosen learning rate: $best_lr"
push "$RUN learning-rate sweep on val (chosen $best_lr)"
if [ "$best_lr" = "none" ]; then
    say "No learning rate passed the selection rule; skipping the final seeds."
    exit 0
fi

# 5-6. Final runs with the chosen rate, and statistics.
final_runs "$best_lr" "$RUN"

say "All done."
