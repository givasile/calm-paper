#!/usr/bin/env bash
# Run the suite in the order the paper needs it, detached, in tmux.
#
#   ./run.sh              five workers, one core each, low priority
#   ./run.sh 3            three workers
#   tmux attach -t calm   watch it;  calm-paper status   see what is left
#
# Safe to run again at any time: finished cells are skipped, and cells left
# half-done by a stopped worker are fitted again.
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${1:-5}"
ORDER="main synthetic eb2m depth sensitivity depth_ladders stability stability_eb2m"
mkdir -p runs/logs
LOG="runs/logs/run-$(date +%Y%m%d-%H%M%S).log"
tmux new-session -d -s calm "nice -n 10 .venv/bin/calm-paper run $ORDER --workers $WORKERS > $LOG 2>&1"
echo "running in tmux session 'calm', log $LOG"
