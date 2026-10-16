#!/usr/bin/env bash
# Run a whole experiment suite, one configuration per process. Resumable: configurations whose result file exists are skipped.
# The configurations are listed in experiments/<group>/configs.txt as "tag|extra arguments".
# Usage: experiments/suite.sh lora      (synthetic text tickets)
#        experiments/suite.sh vision    (synthetic shapes images)
set -u
cd "$(dirname "$0")/.."
group="$1"
case "$group" in lora) script=experiments/lora/run.py ;; vision) script=experiments/vision/run.py ;; *) echo "unknown suite $group"; exit 1 ;; esac
mkdir -p "experiments/results/$group/logs"
while IFS= read -r entry; do
  case "$entry" in ""|\#*) continue ;; esac
  tag="${entry%%|*}"; args="${entry#*|}"
  if [ -f "experiments/results/$group/$tag.json" ]; then echo "skip $tag (done)"; continue; fi
  echo "=== $group / $tag ($(date +%H:%M:%S)) ==="
  # shellcheck disable=SC2086
  .venv/bin/python -u "$script" --tag "$tag" --no-save $args > "experiments/results/$group/logs/$tag.log" 2>&1 || echo "FAILED $tag (see log)"
done < "experiments/$group/configs.txt"
echo "suite $group finished $(date +%H:%M:%S)"
