#!/usr/bin/env bash
# Run one pipeline stage on a Vast.ai instance, push the new artifacts to GitHub, then
# stop the instance so GPU billing ends. Arguments are passed straight to run.py:
#
#   bash scripts/run_then_stop.sh stage=evaluate eval.split=test
#
# The instance is stopped whether the run succeeds or fails. Its disk is kept (and still
# billed) until you destroy it, so nothing is lost if the push fails.
set -u

# Fail instead of waiting for a password prompt that nobody will answer.
export GIT_TERMINAL_PROMPT=0

python run.py "$@"
status=$?
echo "run.py exited with status $status"

git add -A artifacts
if git commit -m "Results: $*"; then
    git push || echo "Push failed: results are still on the instance disk."
fi

command -v vastai > /dev/null || pip install -q vastai
vastai stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
exit $status
