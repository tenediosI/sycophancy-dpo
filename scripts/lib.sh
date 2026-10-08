# Shared helpers for the GPU session scripts (source it; do not run it).
#
# Expects PREFIX, EXTRA and BASE to be set. NO_PUSH=1 disables committing and pushing;
# the instance is only stopped when $CONTAINER_ID is set, i.e. on Vast.

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

# Each step skips work whose output exists and redoes (overwrite=true) a step that was
# interrupted half-way, so rerunning a session script resumes it.
train_dpo() {  # name lr seed [extra overrides...]
    local name=$1 lr=$2 seed=$3
    shift 3
    # The summary is written after the merged model is saved, so it marks a finished run;
    # the checkpoint must also still be on this disk (it is not pushed).
    [ -f "artifacts/results/$name/train/summary.json" ] && [ -d "artifacts/checkpoints/$name/merged" ] \
        && { say "skip training $name (exists)"; return; }
    run stage=dpo seed="$seed" train.learning_rate="$lr" train.run_name="$name" "$@" overwrite=true
}
evaluate() {  # name split
    [ -f "artifacts/results/$1/$2/summary.json" ] && { say "skip $2 eval of $1 (exists)"; return; }
    run stage=evaluate eval.split="$2" eval.run_name="$1" model.checkpoint="artifacts/checkpoints/$1/merged" overwrite=true
}
capability() {  # name
    [ -f "artifacts/results/$1/capability/summary.json" ] && { say "skip capability of $1 (exists)"; return; }
    run stage=capability capability.run_name="$1" model.checkpoint="artifacts/checkpoints/$1/merged" overwrite=true
}
calibrate() {  # name [checkpoint]
    [ -f "artifacts/results/$1/calibration/summary.json" ] && { say "skip calibration of $1 (exists)"; return; }
    if [ -n "${2:-}" ]; then
        run stage=calibrate calibration.run_name="$1" model.checkpoint="$2" overwrite=true
    else
        run stage=calibrate calibration.run_name="$1" overwrite=true
    fi
}
