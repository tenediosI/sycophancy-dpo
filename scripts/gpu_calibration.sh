#!/usr/bin/env bash
# GPU session 6 (Step 8, calibration) on a Vast.ai instance, then push results and stop:
#
#   bash scripts/gpu_calibration.sh 2>&1 | tee -a session.log
#
# The trained models of sessions 2 and 5 were not kept, so they are retrained with the
# same code, data and settings under <name>_rerun. Retraining on a GPU is not bit-exact,
# so each rerun gets its own pushback evaluation on test: its confidence is always joined
# with its own behaviour, and the rerun can be compared with the original (a
# reproducibility check, done locally with stage=analyse).
#
#   1. Base model: calibration on test (its pushback results already exist).
#   2. For each model below: retrain, pushback evaluation on test, calibration on test.
#   3. (HF_REPO set) Upload the first model's LoRA adapter to the Hugging Face Hub, private.
#
# Models, in order (QUICK=1 runs only the first):
#   dpo_bal_ep3_lr1e-4_seed1_rerun  balanced pairs, 3 epochs, lr 1e-4 (session 5, best)
#   dpo_seed1_rerun                 3:1 pairs, 1 epoch, lr 5e-5 (session 2, stubborn)
#   dpo_bal_ep3_lr1e-4_seed2_rerun
#   dpo_bal_ep3_lr1e-4_seed3_rerun
#
# Environment overrides:
#   QUICK=1                 base + the first model only
#   HF_REPO="user/name"     upload the first model's adapter (needs HF_TOKEN set)
#   PREFIX, EXTRA, NO_PUSH  as in gpu_session.sh (for local tests)
set -euo pipefail

PREFIX=${PREFIX:-""}
EXTRA=${EXTRA:-""}
BASE="${PREFIX}base"

source "$(dirname "$0")/lib.sh"

#        name                             lr    seed ratio epochs
MODELS=("dpo_bal_ep3_lr1e-4_seed1_rerun  1e-4  1    1     3"
        "dpo_seed1_rerun                 5e-5  1    3     1"
        "dpo_bal_ep3_lr1e-4_seed2_rerun  1e-4  2    1     3"
        "dpo_bal_ep3_lr1e-4_seed3_rerun  1e-4  3    1     3")
[ -n "${QUICK:-}" ] && MODELS=("${MODELS[0]}")

say "Calibration session: ${#MODELS[@]} model(s), QUICK=${QUICK:-} HF_REPO=${HF_REPO:-}"

# 1. Base model.
calibrate "$BASE"
push "base model calibration"

# 2. Retrained models.
for spec in "${MODELS[@]}"; do
    read -r stem lr seed ratio epochs <<< "$spec"
    name="${PREFIX}${stem}"
    train_dpo "$name" "$lr" "$seed" train.pairs.wrong_per_correct="$ratio" train.num_train_epochs="$epochs"
    evaluate "$name" test
    calibrate "$name" "artifacts/checkpoints/$name/merged"
    push "$name: rerun, test evaluation and calibration"
done

# 3. Adapter upload (optional).
if [ -n "${HF_REPO:-}" ]; then
    read -r stem _ <<< "${MODELS[0]}"
    python scripts/push_adapter.py "artifacts/checkpoints/${PREFIX}${stem}/adapter" "$HF_REPO" "${PREFIX}${stem}" \
        || say "Adapter upload failed; the adapter is still on the instance disk."
fi

say "All done."
