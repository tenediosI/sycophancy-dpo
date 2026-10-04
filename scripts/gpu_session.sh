#!/usr/bin/env bash
# Full GPU session for Steps 4-7 on a Vast.ai instance, then push results and stop:
#
#   bash scripts/gpu_session.sh 2>&1 | tee -a session.log
#
#   1. Regenerate preference pairs (train, val).
#   2. Base model on val (reference for the sweep). Base test results already exist.
#   3. DPO learning-rate sweep, seed 0, each evaluated on val.
#   4. Pick the learning rate with scripts/pick_lr.py (rule fixed in advance, val only).
#   5. DPO with that rate for seeds 1-3, each evaluated on test (pushback + capability).
#
# Results are committed and pushed after every phase, and once more on exit. The instance
# is stopped on exit whether the session succeeds or fails.
#
# Environment overrides (for a quick local test of the script itself):
#   LRS="5e-6 2e-5 5e-5"   learning rates to sweep
#   SEEDS="1 2 3"          final training seeds
#   PREFIX=""              prefix for run names, e.g. "debug/" (debug dirs are gitignored)
#   EXTRA=""               extra Hydra overrides for every run.py call, e.g. "experiment=debug"
#   NO_PUSH=1              do not commit or push
#   SKIP_GENERATE=1        skip step 1 (resume after the pairs were generated and pushed)
#   (the instance is only stopped when $CONTAINER_ID is set, i.e. on Vast)
set -euo pipefail

LRS=${LRS:-"5e-6 2e-5 5e-5"}
SEEDS=${SEEDS:-"1 2 3"}
PREFIX=${PREFIX:-""}
EXTRA=${EXTRA:-""}
BASE="${PREFIX}base"

export GIT_TERMINAL_PROMPT=0

say() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
    say "python run.py $* $EXTRA"
    # shellcheck disable=SC2086  # EXTRA is a list of overrides
    python run.py "$@" $EXTRA
}

push() {
    [ -n "${NO_PUSH:-}" ] && return 0
    git add -A artifacts
    if git commit -q -m "Results: $1"; then
        git push -q || say "Push failed: results are still on the instance disk."
    fi
}

finish() {
    status=$?
    say "Session ended with status $status"
    push "GPU session end (status $status)" || true
    if [ -n "${CONTAINER_ID:-}" ]; then
        command -v vastai > /dev/null || pip install -q vastai
        vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
    fi
}
trap finish EXIT

# 1. Preference pairs. overwrite: the pairs from run 1 used the old selection.
# SKIP_GENERATE=1 resumes a session whose pairs were already generated and pushed.
if [ -z "${SKIP_GENERATE:-}" ]; then
    for split in train val; do
        run stage=generate generate.split="$split" overwrite=true
    done
    push "preference pairs (all pairs saved)"
fi

# 2. Base model references (skipped when they already exist).
[ -f "artifacts/results/$BASE/val/summary.json" ] || run stage=evaluate eval.split=val eval.run_name="$BASE"
[ -f "artifacts/results/$BASE/test/summary.json" ] || run stage=evaluate eval.split=test eval.run_name="$BASE"
[ -f "artifacts/results/$BASE/capability/summary.json" ] || run stage=capability capability.run_name="$BASE"
push "base model references"

# Steps 3 and 5 are resumable: rerunning the script skips work whose output exists and
# redoes (overwrite=true) a step that was interrupted half-way.
train_dpo() {  # name lr seed
    # The summary is written after the merged model is saved, so it marks a finished run;
    # the checkpoint must also still be on this disk (it is not pushed).
    [ -f "artifacts/results/$1/train/summary.json" ] && [ -d "artifacts/checkpoints/$1/merged" ] \
        && { say "skip training $1 (exists)"; return; }
    run stage=dpo seed="$3" train.learning_rate="$2" train.run_name="$1" overwrite=true
}
evaluate() {  # name split
    [ -f "artifacts/results/$1/$2/summary.json" ] && { say "skip $2 eval of $1 (exists)"; return; }
    run stage=evaluate eval.split="$2" eval.run_name="$1" model.checkpoint="artifacts/checkpoints/$1/merged" overwrite=true
}
capability() {  # name
    [ -f "artifacts/results/$1/capability/summary.json" ] && { say "skip capability of $1 (exists)"; return; }
    run stage=capability capability.run_name="$1" model.checkpoint="artifacts/checkpoints/$1/merged" overwrite=true
}

# 3. Learning-rate sweep on val, seed 0.
sweep=()
for lr in $LRS; do
    name="${PREFIX}dpo_lr${lr}_seed0"
    train_dpo "$name" "$lr" 0
    evaluate "$name" val
    sweep+=("$lr=$name")
done

# 4. Choose the learning rate (last line of the output).
best_lr=$(python scripts/pick_lr.py "$BASE" "${sweep[@]}" | tee /dev/stderr | tail -n 1)
say "Chosen learning rate: $best_lr"
push "DPO learning-rate sweep on val (chosen $best_lr)"

# 5. Final runs: fresh seeds, evaluated on test.
for seed in $SEEDS; do
    name="${PREFIX}dpo_seed${seed}"
    train_dpo "$name" "$best_lr" "$seed"
    evaluate "$name" test
    capability "$name"
    push "DPO seed $seed (lr $best_lr) on test"
done

say "All done."
